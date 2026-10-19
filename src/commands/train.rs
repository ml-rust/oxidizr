//! The `train` command: its flags, the run directory and record, and the launch of every rank.
//!
//! The run plan resolves the config, the GPU ids and the resume once. Each
//! rank then trains in [`crate::commands::train_rank`]. One GPU, or the CPU,
//! is one rank on the calling thread.

use crate::commands::train_plan::{Launch, RunPlan};
use crate::commands::train_rank::{train_rank, RankRun};
use crate::config::LitConfig;
use crate::dp::{run_ranks, RankDevice, RankGroup};
use crate::run_record::{RunProgress, RunRecord};
use anyhow::{Context, Result};
use boostr::ops::{FusedOptimizerOps, TensorOps};
use boostr::runtime::RuntimeClient;
use clap::Args;
use std::path::Path;
use std::sync::{Mutex, PoisonError};

pub use crate::commands::train_report::estimate_params;

/// Training command arguments
#[derive(Args, Debug)]
pub struct TrainArgs {
    /// Path to the configuration YAML file
    #[arg(short = 'f', long = "config", default_value = "models/nano.yaml")]
    pub config_file: String,

    /// Path to training data file (memory-mapped binary token file)
    #[arg(short = 'd', long = "data")]
    pub data_file: Option<String>,

    /// Override target device: "gpu" or "cpu" (default from config, fallback to gpu)
    #[arg(long = "target-device")]
    pub target_device: Option<String>,

    /// Override sequence length from YAML config
    #[arg(long = "seq-len")]
    pub seq_length: Option<usize>,

    /// Override the per-rank batch size from YAML config. The global batch
    /// is this times the GPU count times gradient accumulation.
    #[arg(long = "batch-size")]
    pub batch_size: Option<usize>,

    /// Override gradient accumulation from YAML config
    #[arg(long = "grad-accum")]
    pub gradient_accumulation: Option<usize>,

    /// Headless mode: output only JSON metrics (no fancy UI)
    #[arg(long)]
    pub headless: bool,

    /// Override max training steps from YAML config
    #[arg(long = "max-steps")]
    pub max_steps: Option<usize>,

    /// Resume training from a checkpoint file (.safetensors) or "auto" to find latest
    #[arg(long = "resume")]
    pub resume: Option<String>,

    /// GPU ids, one data-parallel rank per id, for example `0,1`. More than
    /// one id needs `--target-device gpu` and a build with `--features cuda,nccl`.
    #[arg(long = "gpus")]
    pub gpus: Option<String>,

    /// Sync backend name: "single", "cpu", or "nccl". Overrides
    /// `trainer.distributed.backend` only when given. NCCL is the only
    /// multi-GPU backend.
    #[arg(long = "sync-backend")]
    pub sync_backend: Option<String>,

    /// Enable async data prefetching (CPU-GPU overlap)
    /// Prefetches N batches in background thread while GPU trains
    #[arg(long = "prefetch", default_value = "0")]
    pub prefetch: usize,

    /// Maximum checkpoints to keep (0 = unlimited, default = 10)
    #[arg(long = "max-checkpoints")]
    pub max_checkpoints: Option<usize>,

    /// Commit checkpoint sets on a background thread. Sets
    /// `trainer.async_checkpoint` to true. Absent, the config value applies.
    #[arg(long = "async-checkpoint")]
    pub async_checkpoint: bool,

    /// Model precision override: f32, f16, or bf16. Overrides `model.dtype`
    /// only when given; the config value applies otherwise.
    #[arg(long = "dtype")]
    pub dtype: Option<String>,

    /// Checkpoint directory override. Overrides `trainer.checkpoint_dir`
    /// only when given (also expanding any `${VAR}` placeholder it names);
    /// the config value applies otherwise.
    #[arg(long = "checkpoint-dir")]
    pub checkpoint_dir: Option<String>,

    /// Run directory for run.json, steps.json and evals.json. Default: $OXIDIZED_RUN_DIR,
    /// else ./runs/<timestamp>. The directory is created when absent.
    #[arg(long = "run-dir")]
    pub run_dir: Option<String>,
}

/// Run the training command
pub fn run_train(args: TrainArgs) -> Result<()> {
    let config = LitConfig::from_yaml(&args.config_file)?;

    #[cfg(feature = "cuda")]
    {
        let use_cpu = args
            .target_device
            .as_deref()
            .map(|dev| dev.eq_ignore_ascii_case("cpu"))
            .unwrap_or(config.trainer.target_device == crate::config::TargetDevice::Cpu);
        if !use_cpu {
            let launch = Launch::Cli { uses_cpu: false };
            return run_train_with_runtime::<boostr::CudaRuntime>(args, config, None, launch);
        }
    }

    let launch = Launch::Cli { uses_cpu: true };
    run_train_with_runtime::<boostr::CpuRuntime>(args, config, None, launch)
}

/// Runs the training command on the ranks of `group`, one thread per rank.
///
/// Each rank opens its own device: its `--gpus` id, or the runtime default
/// without a GPU list. A GPU list must name one GPU per rank. The build and
/// target checks of `--gpus` do not apply, so tests can run several CPU ranks
/// over `RankGroup::local`. Rank 0 writes the run directory and checkpoints.
///
/// # Errors
/// Returns the error of the rank that failed first, after `run.json` records it.
pub fn run_train_with_group<R>(args: TrainArgs, group: RankGroup) -> Result<()>
where
    R: RankDevice,
    R::Client: RuntimeClient<R>
        + TensorOps<R>
        + FusedOptimizerOps<R>
        + boostr::RandomOps<R>
        + boostr::ScalarOps<R>
        + boostr::BinaryOps<R>
        + boostr::CompareOps<R>
        + boostr::ConvOps<R>
        + boostr::FlashAttentionOps<R>
        + boostr::IndexingOps<R>
        + boostr::NormalizationOps<R>
        + boostr::QuantMatmulOps<R>
        + boostr::RoPEOps<R>
        + boostr::TypeConversionOps<R>
        + boostr::DequantOps<R>
        + boostr::GatedDeltaNetOps<R>,
{
    let config = LitConfig::from_yaml(&args.config_file)?;
    let launch = Launch::Group {
        world: group.world_size(),
    };
    run_train_with_runtime::<R>(args, config, Some(group), launch)
}

/// The run directory: `--run-dir`, else `$OXIDIZED_RUN_DIR`, else `./runs/<timestamp>`.
fn resolve_run_dir(args: &TrainArgs) -> Result<String> {
    let dir = match (&args.run_dir, std::env::var("OXIDIZED_RUN_DIR")) {
        (Some(dir), _) => dir.clone(),
        (None, Ok(dir)) => dir,
        (None, Err(_)) => format!("./runs/{}", chrono::Local::now().format("%Y%m%d_%H%M%S")),
    };
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create run directory '{dir}'"))?;
    Ok(dir)
}

/// Runs one training run and writes its `run.json`, whatever the outcome.
///
/// The record goes to the run directory before the result returns, so `main`
/// exits only after it is on disk. Rank 0 fills the record.
fn run_train_with_runtime<R>(
    args: TrainArgs,
    config: LitConfig,
    group: Option<RankGroup>,
    launch: Launch,
) -> Result<()>
where
    R: RankDevice,
    R::Client: RuntimeClient<R>
        + TensorOps<R>
        + FusedOptimizerOps<R>
        + boostr::RandomOps<R>
        + boostr::ScalarOps<R>
        + boostr::BinaryOps<R>
        + boostr::CompareOps<R>
        + boostr::ConvOps<R>
        + boostr::FlashAttentionOps<R>
        + boostr::IndexingOps<R>
        + boostr::NormalizationOps<R>
        + boostr::QuantMatmulOps<R>
        + boostr::RoPEOps<R>
        + boostr::TypeConversionOps<R>
        + boostr::DequantOps<R>
        + boostr::GatedDeltaNetOps<R>,
{
    let run_dir = resolve_run_dir(&args)?;
    let progress = Mutex::new(RunProgress::start(&run_dir, R::name()));
    let result = train_in_run_dir::<R>(&args, config, &run_dir, &progress, group, launch);
    let progress = progress
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner);
    match progress.finish(&result).write(Path::new(&run_dir)) {
        Ok(()) if !args.headless => println!(
            "Run metadata saved to: {}",
            RunRecord::path_in(Path::new(&run_dir)).display()
        ),
        Ok(()) => {}
        // The training result outranks its record: warn, and keep the result.
        Err(e) => eprintln!("Warning: {e:#}"),
    }
    result
}

/// Plans the run, opens each rank's device, and trains every rank.
fn train_in_run_dir<R>(
    args: &TrainArgs,
    config: LitConfig,
    run_dir: &str,
    progress: &Mutex<RunProgress>,
    group: Option<RankGroup>,
    launch: Launch,
) -> Result<()>
where
    R: RankDevice,
    R::Client: RuntimeClient<R>
        + TensorOps<R>
        + FusedOptimizerOps<R>
        + boostr::RandomOps<R>
        + boostr::ScalarOps<R>
        + boostr::BinaryOps<R>
        + boostr::CompareOps<R>
        + boostr::ConvOps<R>
        + boostr::FlashAttentionOps<R>
        + boostr::IndexingOps<R>
        + boostr::NormalizationOps<R>
        + boostr::QuantMatmulOps<R>
        + boostr::RoPEOps<R>
        + boostr::TypeConversionOps<R>
        + boostr::DequantOps<R>
        + boostr::GatedDeltaNetOps<R>,
{
    let plan = RunPlan::build(args, config, run_dir, launch)?;
    progress
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .final_step = plan.resume.start_step;
    // Every device opens before any rank starts. NCCL init needs every client.
    let devices = (0..plan.world)
        .map(|rank| R::rank_device(plan.gpu_id(rank)))
        .collect::<Result<Vec<_>>>()?;
    let group = match group {
        Some(group) => group,
        None => {
            let clients: Vec<R::Client> = devices.iter().map(|(_, c)| c.clone()).collect();
            R::rank_group(&clients)?
        }
    };
    run_ranks(&group, |slot| {
        let Some((device, client)) = devices.get(slot.rank).cloned() else {
            anyhow::bail!(
                "rank {} has no device: {} devices opened for {} ranks",
                slot.rank,
                devices.len(),
                slot.world
            );
        };
        let progress = (slot.rank == 0).then_some(progress);
        train_rank::<R>(RankRun {
            plan: &plan,
            slot,
            device,
            client,
            progress,
        })
    })
    .map(drop)
}
