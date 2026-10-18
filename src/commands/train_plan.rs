//! The rank-independent part of a training run, resolved once before any rank starts.

use crate::checkpoint_set::{plan_optimizer_resume, OptimizerResume};
use crate::commands::overrides::Source;
use crate::commands::train_data::TokenSource;
use crate::commands::train_overrides::apply_cli_overrides;
use crate::commands::train_resume::{resolve_resume, ResumePoint};
use crate::commands::TrainArgs;
use crate::config::{DTypePrecision, LitConfig};
use crate::distributed::{
    check_gpu_ids, check_sync_backend, parse_gpu_ids, reject_duplicate_gpus, BuildSupport,
};
use crate::dp::check_resume_world;
use crate::loss_scale::{read_loss_scale_state, resolve_loss_scale};
use crate::metrics::{read_spike_guard_state, SpikeGuardState};
use crate::weight_decay::{inherited_weight_decay, resolve_weight_decay, WeightDecaySpec};
use anyhow::Result;
use boostr::optimizer::GradScalerState;

/// How the ranks of a run come to exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Launch {
    /// The CLI: one rank per GPU id. `uses_cpu` is true for a CPU run.
    Cli { uses_cpu: bool },
    /// A caller-built group of `world` ranks, for example in-process CPU ranks.
    Group { world: usize },
}

/// Everything every rank of a run shares: the final config, the resume point, the data.
pub struct RunPlan {
    /// The config after every CLI override.
    pub config: LitConfig,
    pub data_file: Option<String>,
    pub headless: bool,
    pub prefetch: usize,
    pub run_dir: String,
    /// One GPU id per rank. Empty: every rank uses the runtime's default device.
    pub gpu_ids: Vec<usize>,
    /// Data-parallel ranks.
    pub world: usize,
    pub resume: ResumePoint,
    pub optimizer_resume: OptimizerResume,
    pub loss_scale_state: Option<GradScalerState>,
    /// The spike guard state of the resumed checkpoint. `None` for a fresh run.
    pub spike_guard_state: Option<SpikeGuardState>,
    pub weight_decay: WeightDecaySpec,
    /// The training tokens. Generated tokens are drawn once, so every rank reads the same.
    pub source: TokenSource,
}

impl RunPlan {
    /// The GPU id of `rank`. `None` without a GPU list.
    pub fn gpu_id(&self, rank: usize) -> Option<usize> {
        self.gpu_ids.get(rank).copied()
    }

    /// Applies `args` to `config`, checks the GPU ids, and resolves the resume.
    ///
    /// Prints, prompts and checkpoint clearing happen here, once, on the calling thread.
    ///
    /// # Errors
    /// Returns an error for a bad flag, an unsupported GPU list, a resume
    /// whose world size differs, or an unusable checkpoint.
    pub fn build(
        args: &TrainArgs,
        mut config: LitConfig,
        run_dir: &str,
        launch: Launch,
    ) -> Result<Self> {
        let headless = args.headless;
        if !headless {
            println!("=== Oxidizr Training Framework ===");
            println!("Run directory: {}\n", run_dir);
            println!("Loading configuration from: {}", args.config_file);
        }
        let backend_source = apply_cli_overrides(args, &mut config)?;
        let (gpu_ids, world) = resolve_ranks(args, &mut config, launch, backend_source)?;

        if config.model.dtype != DTypePrecision::F32 && !headless {
            println!(
                "Using {} precision (RmsNorm/SSM ops use F32 internally for stability)",
                config.model.dtype
            );
        }
        // The dtype is final here, so a block under bf16 or f32 fails before the model builds.
        let loss_scale =
            resolve_loss_scale(config.trainer.loss_scale.as_ref(), config.model.dtype)?;
        if let (Some(scale), false) = (loss_scale, headless) {
            println!(
                "Loss scaling on: initial {}, growth {}x every {} clean steps, backoff {}x",
                scale.initial, scale.growth_factor, scale.growth_interval, scale.backoff_factor
            );
        }
        if !headless {
            println!("Configuration loaded successfully!\n");
        }

        let resume = resolve_resume(
            args.resume.as_deref(),
            &config,
            args.data_file.as_deref(),
            headless,
        )?;
        let (start_step, meta) = (resume.start_step, resume.meta_path.as_deref());
        let global_batch_rows = config.trainer.batch_size.saturating_mul(world);
        if let Some(warning) = check_resume_world(meta, start_step, world, global_batch_rows)? {
            eprintln!("{warning}");
        }
        // Plan the optimizer restore before the long setup, so a missing file fails early.
        let optimizer_resume = plan_optimizer_resume(resume.weights.as_deref(), meta, start_step)?;
        // The config's weight decay rule, or the rule of the checkpoint this run resumes.
        let inherited_decay = inherited_weight_decay(meta, start_step)?;
        let loss_scale_state = read_loss_scale_state(meta, start_step)?;
        let spike_guard_state = read_spike_guard_state(meta, start_step)?;
        let weight_decay =
            resolve_weight_decay(&config.trainer, inherited_decay.as_ref(), headless)?;

        if !headless {
            match &args.data_file {
                Some(path) => println!("Loading training data from memory-mapped file: {path}"),
                None => println!("Creating dummy training data..."),
            }
        }
        let source = TokenSource::for_run(args.data_file.as_deref(), &config);
        Ok(Self {
            config,
            data_file: args.data_file.clone(),
            headless,
            prefetch: args.prefetch,
            run_dir: run_dir.to_string(),
            gpu_ids,
            world,
            resume,
            optimizer_resume,
            loss_scale_state,
            spike_guard_state,
            weight_decay,
            source,
        })
    }
}

/// The GPU id list and world size of a run, after every check.
///
/// `--gpus` overrides `trainer.distributed.gpu_ids`. A CLI launch runs one
/// rank per id and checks that this build and target support them. A group
/// launch takes the group's world size. A GPU list must then name one GPU per rank.
fn resolve_ranks(
    args: &TrainArgs,
    config: &mut LitConfig,
    launch: Launch,
    backend_source: Source,
) -> Result<(Vec<usize>, usize)> {
    let input = match &args.gpus {
        Some(raw) => {
            config.trainer.distributed.gpu_ids = parse_gpu_ids(raw)?;
            format!("--gpus {raw}")
        }
        None => format!(
            "trainer.distributed.gpu_ids {:?}",
            config.trainer.distributed.gpu_ids
        ),
    };
    let gpu_ids = config.trainer.distributed.gpu_ids.clone();
    reject_duplicate_gpus(&input, &gpu_ids)?;
    let world = match launch {
        Launch::Cli { uses_cpu } => {
            check_gpu_ids(&input, &gpu_ids, uses_cpu, BuildSupport::current())?;
            gpu_ids.len().max(1)
        }
        Launch::Group { world } => {
            if !gpu_ids.is_empty() && gpu_ids.len() != world {
                anyhow::bail!(
                    "`{input}` names {} GPUs, and the rank group has {world} ranks. Name one GPU \
                     per rank, or remove the GPU list.",
                    gpu_ids.len()
                );
            }
            world
        }
    };
    check_sync_backend(
        &config.trainer.distributed.backend,
        backend_source == Source::Cli,
        world,
    )?;
    Ok((gpu_ids, world))
}
