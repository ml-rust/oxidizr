//! Training command implementation
//!
//! This module contains the training logic, extracted from the original main.rs.

use crate::prelude::{DType, Tensor, VarMap};
use crate::run_record::{RunProgress, RunRecord, STEPS_FILE};
use anyhow::{Context, Result};
use boostr::autograd::Var;
use boostr::ops::{FusedOptimizerOps, TensorOps};
use boostr::runtime::{Runtime, RuntimeClient};
use boostr::tensor::TensorId;
use clap::Args;
use std::collections::HashMap;
use std::path::Path;

use crate::checkpoint_set::{manual_resume_target, plan_optimizer_resume};
use crate::commands::overrides::{
    resolve_checkpoint_dir, resolve_dtype, resolve_sync_backend, Source,
};
use crate::config::{
    find_latest_checkpoint, CheckpointMeta, DTypePrecision, LitConfig, OxidizrModelConfig,
};
use crate::data::{
    create_dummy_data, DataPosition, LitDataLoader, PrefetchDataLoader, RowLayout, RowOrder,
};
use crate::model::LitGPT;
use crate::trainer::{LitTrainer, TrainerRunConfig};

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

    /// Override batch size from YAML config
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

    /// GPU IDs for multi-GPU training (e.g., "0,1,2,3")
    #[arg(long = "gpus")]
    pub gpus: Option<String>,

    /// Gradient synchronization backend override: "single", "cpu", or "nccl".
    /// Overrides `trainer.distributed.backend` only when given; the config
    /// value applies otherwise.
    #[arg(long = "sync-backend")]
    pub sync_backend: Option<String>,

    /// Enable async data prefetching (CPU-GPU overlap)
    /// Prefetches N batches in background thread while GPU trains
    #[arg(long = "prefetch", default_value = "0")]
    pub prefetch: usize,

    /// Maximum checkpoints to keep (0 = unlimited, default = 10)
    #[arg(long = "max-checkpoints")]
    pub max_checkpoints: Option<usize>,

    /// Model precision override: f32, f16, or bf16. Overrides `model.dtype`
    /// only when given; the config value applies otherwise.
    #[arg(long = "dtype")]
    pub dtype: Option<String>,

    /// Checkpoint directory override. Overrides `trainer.checkpoint_dir`
    /// only when given (also expanding any `${VAR}` placeholder it names);
    /// the config value applies otherwise.
    #[arg(long = "checkpoint-dir")]
    pub checkpoint_dir: Option<String>,

    /// Run directory for run.json and steps.json. Default: $OXIDIZED_RUN_DIR,
    /// else ./runs/<timestamp>. The directory is created when absent.
    #[arg(long = "run-dir")]
    pub run_dir: Option<String>,
}

/// Parse GPU IDs from comma-separated string (e.g., "0,1,2,3")
fn parse_gpu_ids(gpus_str: &Option<String>) -> Vec<usize> {
    match gpus_str {
        Some(s) => s
            .split(',')
            .filter_map(|id| id.trim().parse::<usize>().ok())
            .collect(),
        None => vec![0], // Default to GPU 0
    }
}

/// Prompt user for yes/no confirmation
/// Returns true if user confirms (y/Y), false otherwise
fn prompt_confirmation(message: &str) -> bool {
    use std::io::{self, Write};

    print!("{} [y/N]: ", message);
    // A flush error leaves the prompt unshown. The answer is still read, so it is ignored.
    let _ = io::stdout().flush();

    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_err() {
        return false;
    }

    matches!(input.trim().to_lowercase().as_str(), "y" | "yes")
}

/// Clear all checkpoint files from a directory.
///
/// Removes weights and optimizer state (`.safetensors`), metas (`.meta.json`),
/// `config.json`, and temp files that an interrupted atomic write left behind.
fn clear_checkpoint_directory(checkpoint_dir: &str) -> Result<usize> {
    use std::path::Path;

    let dir = Path::new(checkpoint_dir);
    if !dir.exists() {
        return Ok(0);
    }

    let mut removed_count = 0;

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
            // `.safetensors` also matches `.optimizer.safetensors`. A temp file is
            // `.<file_name>.tmp-<pid>-<n>`.
            let is_temp = filename.starts_with('.') && filename.contains(".tmp-");
            if filename.ends_with(".safetensors")
                || filename.ends_with(".meta.json")
                || filename == "config.json"
                || is_temp
            {
                if let Err(e) = std::fs::remove_file(&path) {
                    eprintln!("Warning: Failed to remove {}: {}", path.display(), e);
                } else {
                    removed_count += 1;
                }
            }
        }
    }

    Ok(removed_count)
}

/// Loader position for a run that resumes at `start_step`.
///
/// Reads `data_position` from the checkpoint metadata at `meta_path`.
/// Metadata written before row loading carries none, and an explicit
/// `--resume <path>` can have no metadata at all. The position then follows
/// from `start_step` under the current row layout and accumulation, and a
/// stderr line says so.
fn resume_data_position(
    meta_path: Option<&str>,
    start_step: usize,
    layout: &RowLayout,
    grad_accum: usize,
) -> Result<DataPosition> {
    if let Some(path) = meta_path {
        let meta = CheckpointMeta::load(path).map_err(|e| {
            anyhow::anyhow!(
                "failed to read checkpoint metadata {path}: {e}. Restore the file, or pass \
                 the checkpoint's .safetensors path to --resume without it."
            )
        })?;
        if let Some(position) = meta.data_position {
            return Ok(position);
        }
    }
    let position = layout.position_after_steps(start_step, grad_accum);
    if start_step > 0 {
        eprintln!(
            "Checkpoint metadata ({}) records no data position: resuming the data stream at \
             epoch {} row {}, derived from step {start_step} x {grad_accum} batches per step \
             x {} rows per batch.",
            meta_path.unwrap_or("none found"),
            position.epoch,
            position.row,
            layout.batch_size()
        );
    }
    Ok(position)
}

/// Get CUDA memory info in a human-readable format
fn get_cuda_memory_info<R: Runtime>(_device: &R::Device) -> Result<String> {
    // Just query all GPUs - simpler and more reliable
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output();

    if let Ok(output) = output {
        if output.status.success() {
            let stats = String::from_utf8_lossy(&output.stdout);
            // Take the first line (first GPU)
            if let Some(line) = stats.lines().next() {
                let parts: Vec<&str> = line.trim().split(',').collect();
                if parts.len() == 3 {
                    let total_mb: f32 = parts[0].trim().parse().unwrap_or(0.0);
                    let used_mb: f32 = parts[1].trim().parse().unwrap_or(0.0);
                    let free_mb: f32 = parts[2].trim().parse().unwrap_or(0.0);
                    return Ok(format!(
                        "Total: {:.2} GB, Used: {:.2} GB, Free: {:.2} GB",
                        total_mb / 1024.0,
                        used_mb / 1024.0,
                        free_mb / 1024.0
                    ));
                }
            }
        }
    }
    Ok("Unable to query memory info".to_string())
}

/// Estimate the number of parameters in the model
pub fn estimate_params(cfg: &OxidizrModelConfig) -> String {
    // Embedding: vocab_size * hidden_size
    let embed_params = cfg.vocab_size * cfg.hidden_size;

    // Each layer:
    // - Attention: Q + K + V + O projections (accounting for GQA)
    //   - Q: hidden * hidden
    //   - K: hidden * (kv_heads * head_dim)
    //   - V: hidden * (kv_heads * head_dim)
    //   - O: hidden * hidden
    // - MLP: gate_proj + up_proj + down_proj
    //   - gate: hidden * (4*hidden)
    //   - up: hidden * (4*hidden)
    //   - down: (4*hidden) * hidden
    // - LayerNorms: 2 * hidden (negligible)
    let head_dim = cfg.hidden_size / cfg.num_heads;
    let kv_heads = cfg.kv_heads.unwrap_or(cfg.num_heads);
    let kv_size = kv_heads * head_dim;
    let intermediate_size = cfg.intermediate_size.unwrap_or(cfg.hidden_size * 4);

    let attention_params = cfg.hidden_size * cfg.hidden_size + // Q proj
                          cfg.hidden_size * kv_size +          // K proj
                          cfg.hidden_size * kv_size +          // V proj
                          cfg.hidden_size * cfg.hidden_size; // O proj

    let mlp_params = cfg.hidden_size * intermediate_size +  // gate_proj
                    cfg.hidden_size * intermediate_size +   // up_proj
                    intermediate_size * cfg.hidden_size; // down_proj

    let layernorm_params = 2 * cfg.hidden_size; // 2 RMSNorms per layer

    let layer_params = cfg.num_layers * (attention_params + mlp_params + layernorm_params);

    // LM Head: hidden_size * vocab_size, but zero when tied — a tied model
    // reuses the embedding matrix rather than holding a second one, so
    // counting it again overstates the model by a whole vocab-sized matrix.
    let lm_head_params = if cfg.tie_word_embeddings {
        0
    } else {
        cfg.hidden_size * cfg.vocab_size
    };

    // Final norm
    let final_norm_params = cfg.hidden_size;

    let total = embed_params + layer_params + lm_head_params + final_norm_params;

    if total > 1_000_000_000 {
        format!("{:.2}B", total as f64 / 1_000_000_000.0)
    } else if total > 1_000_000 {
        format!("{:.2}M", total as f64 / 1_000_000.0)
    } else {
        format!("{}K", total / 1000)
    }
}

fn parameter_tensor_map<R>(params: Vec<(TensorId, &Var<R>)>) -> HashMap<TensorId, Tensor<R>>
where
    R: Runtime<DType = DType>,
{
    params
        .into_iter()
        .map(|(id, var)| (id, var.tensor().clone()))
        .collect()
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
            return run_train_with_runtime::<boostr::CudaRuntime>(args, config);
        }
    }

    run_train_with_runtime::<boostr::CpuRuntime>(args, config)
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
/// exits only after it is on disk.
fn run_train_with_runtime<R>(args: TrainArgs, config: LitConfig) -> Result<()>
where
    R: Runtime<DType = DType> + crate::distributed::RuntimeDeviceFromOrdinal,
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
        + boostr::DequantOps<R>,
{
    let run_dir = resolve_run_dir(&args)?;
    let headless = args.headless;
    let mut progress = RunProgress::start(&run_dir, R::name());
    let result = train_in_run_dir::<R>(args, config, &run_dir, &mut progress);
    match progress.finish(&result).write(Path::new(&run_dir)) {
        Ok(()) if !headless => println!(
            "Run metadata saved to: {}",
            RunRecord::path_in(Path::new(&run_dir)).display()
        ),
        Ok(()) => {}
        // The training result outranks its record: warn, and keep the result.
        Err(e) => eprintln!("Warning: {e:#}"),
    }
    result
}

fn train_in_run_dir<R>(
    args: TrainArgs,
    mut config: LitConfig,
    run_dir: &str,
    progress: &mut RunProgress,
) -> Result<()>
where
    R: Runtime<DType = DType> + crate::distributed::RuntimeDeviceFromOrdinal,
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
        + boostr::DequantOps<R>,
{
    // Apply headless mode flag
    let headless = args.headless;

    if !headless {
        println!("=== Oxidizr Training Framework ===");
        println!("Run directory: {}\n", run_dir);
    }

    // 1. Configuration was loaded before runtime dispatch.
    if !headless {
        println!("Loading configuration from: {}", args.config_file);
    }

    // Apply CLI overrides
    if let Some(seq_len) = args.seq_length {
        if !headless {
            println!(
                "CLI override: seq_length = {} (was {})",
                seq_len, config.model.max_seq_len
            );
        }
        config.model.max_seq_len = seq_len;
    }
    if let Some(batch_size) = args.batch_size {
        if !headless {
            println!(
                "CLI override: batch_size = {} (was {})",
                batch_size, config.trainer.batch_size
            );
        }
        config.trainer.batch_size = batch_size;
    }
    if let Some(grad_accum) = args.gradient_accumulation {
        if !headless {
            println!(
                "CLI override: gradient_accumulation = {} (was {})",
                grad_accum, config.trainer.gradient_accumulation
            );
        }
        config.trainer.gradient_accumulation = grad_accum;
    }
    if let Some(max_steps) = args.max_steps {
        if !headless {
            println!(
                "CLI override: max_steps = {} (was {})",
                max_steps, config.trainer.max_steps
            );
        }
        config.trainer.max_steps = max_steps;
    }

    // Apply distributed training config from CLI
    let gpu_ids = parse_gpu_ids(&args.gpus);
    if gpu_ids.len() > 1 {
        config.trainer.distributed.enabled = true;
        config.trainer.distributed.gpu_ids = gpu_ids.clone();
        let current_backend = config.trainer.distributed.backend.clone();
        match resolve_sync_backend(args.sync_backend.as_deref(), current_backend) {
            Ok((backend, source)) => {
                if !headless {
                    println!(
                        "CLI override: distributed training enabled with {} GPUs: {:?}",
                        gpu_ids.len(),
                        gpu_ids
                    );
                    match source {
                        Source::Cli => println!(
                            "sync_backend = {:?} (source: --sync-backend CLI flag)",
                            backend
                        ),
                        Source::Config => println!(
                            "sync_backend = {:?} (source: trainer.distributed.backend in config)",
                            backend
                        ),
                    }
                }
                config.trainer.distributed.backend = backend;
            }
            Err(e) => {
                eprintln!(
                    "Warning: {} — using config default ({:?})",
                    e, config.trainer.distributed.backend
                );
            }
        }
    } else if args.gpus.is_some() {
        // Single GPU explicitly specified
        config.trainer.distributed.gpu_ids = gpu_ids.clone();
    }

    // Apply max_checkpoints override
    if let Some(max_ckpt) = args.max_checkpoints {
        if !headless {
            println!(
                "CLI override: max_checkpoints = {} (was {})",
                max_ckpt, config.trainer.max_checkpoints
            );
        }
        config.trainer.max_checkpoints = max_ckpt;
    }

    // Apply dtype override: `--dtype` wins only when given, model.dtype from
    // the YAML applies otherwise.
    match resolve_dtype(args.dtype.as_deref(), config.model.dtype) {
        Ok((dtype, source)) => {
            if !headless {
                match source {
                    Source::Cli => println!(
                        "dtype = {} (source: --dtype CLI flag, config had {})",
                        dtype, config.model.dtype
                    ),
                    Source::Config => println!("dtype = {} (source: model.dtype in config)", dtype),
                }
            }
            config.model.dtype = dtype;
        }
        Err(e) => {
            eprintln!(
                "Warning: {e}, using config default ({})",
                config.model.dtype
            );
        }
    }

    // Apply checkpoint_dir override: `--checkpoint-dir` wins only when
    // given (and is itself run through `${VAR}` expansion), the config's
    // already-expanded `trainer.checkpoint_dir` applies otherwise.
    match resolve_checkpoint_dir(
        args.checkpoint_dir.as_deref(),
        &config.trainer.checkpoint_dir,
    ) {
        Ok((checkpoint_dir, source)) => {
            if !headless {
                match source {
                    Source::Cli => println!(
                        "checkpoint_dir = {} (source: --checkpoint-dir CLI flag, config had {})",
                        checkpoint_dir, config.trainer.checkpoint_dir
                    ),
                    Source::Config => println!(
                        "checkpoint_dir = {} (source: trainer.checkpoint_dir in config)",
                        checkpoint_dir
                    ),
                }
            }
            config.trainer.checkpoint_dir = checkpoint_dir;
        }
        Err(e) => {
            anyhow::bail!("invalid --checkpoint-dir: {e}");
        }
    }

    // Show dtype info (BF16/F16 use less memory, F32 is default)
    if config.model.dtype != DTypePrecision::F32 && !headless {
        println!(
            "Using {} precision (RmsNorm/SSM ops use F32 internally for stability)",
            config.model.dtype
        );
    }

    if !headless {
        println!("Configuration loaded successfully!\n");
    }

    // 2. Setup runtime client/device selected by run_train() dispatch.
    let device = R::default_device();
    let client = R::default_client(&device);
    if !headless {
        println!("Using runtime: {}", R::name());
    }

    // 3. Initialize Model
    if !headless {
        println!(
            "Initializing model with {} parameters...",
            estimate_params(&config.model)
        );
    }

    // Model migration owns initialization through boostr VarBuilder/Init and must
    // expose trainable parameters with stable TensorIds.
    let mut varmap = VarMap::<R>::new();
    let mut model = LitGPT::<R>::new_seeded(&config, &mut varmap, &device, &client)?;

    // Load checkpoint if resuming. The metadata path, when there is one,
    // carries the loader position and names the optimizer state file.
    let (start_step, resume_meta_path, resume_weights) = if let Some(ref resume_arg) = args.resume {
        if resume_arg == "auto" {
            // Auto-resume: find latest checkpoint and validate
            if !headless {
                println!(
                    "Auto-resume: searching for latest checkpoint in {}",
                    config.trainer.checkpoint_dir
                );
            }

            match find_latest_checkpoint(&config.trainer.checkpoint_dir)? {
                Some((checkpoint_path, meta_path, step)) => {
                    if !headless {
                        println!("Found checkpoint: {} (step {})", checkpoint_path, step);
                    }

                    // Validate checkpoint matches current config and data
                    if let Some(ref data_path) = args.data_file {
                        let meta = CheckpointMeta::load(&meta_path)?;
                        let validation = meta.validate(&config.model, data_path)?;

                        if !validation.is_valid() {
                            // Show mismatch details
                            println!("\n⚠️  Checkpoint mismatch detected!");
                            println!(
                                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                            );
                            if !validation.model_matches {
                                println!("❌ Model config mismatch:");
                                println!(
                                    "   Checkpoint hash: {}",
                                    validation.checkpoint_model_hash
                                );
                                println!("   Current hash:    {}", validation.current_model_hash);
                            }
                            if !validation.data_matches {
                                println!("❌ Data file mismatch:");
                                println!("   Checkpoint hash: {}", validation.checkpoint_data_hash);
                                println!("   Current hash:    {}", validation.current_data_hash);
                            }
                            println!(
                                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                            );
                            println!("\nThe current configuration does not match the existing checkpoint.");
                            println!(
                                "Starting fresh training will REMOVE all existing checkpoints in:"
                            );
                            println!("  {}", config.trainer.checkpoint_dir);
                            println!();

                            if headless {
                                // In headless mode, fail immediately (no interactive prompt)
                                return Err(anyhow::anyhow!(
                                    "Checkpoint validation failed in headless mode. \
                                     Clear checkpoints manually or fix config mismatch."
                                ));
                            }

                            // Ask user for confirmation
                            if prompt_confirmation(
                                "Do you want to clear old checkpoints and start fresh?",
                            ) {
                                // Clear old checkpoints
                                match clear_checkpoint_directory(&config.trainer.checkpoint_dir) {
                                    Ok(count) => {
                                        println!("🗑️  Removed {} checkpoint file(s)", count);
                                        println!("Starting fresh training from step 0\n");
                                    }
                                    Err(e) => {
                                        eprintln!("Error clearing checkpoints: {}", e);
                                        return Err(anyhow::anyhow!(
                                            "Failed to clear checkpoint directory"
                                        ));
                                    }
                                }
                                (0, None, None) // Start from step 0
                            } else {
                                println!("Aborted. Please resolve the configuration mismatch or use a different checkpoint directory.");
                                return Err(anyhow::anyhow!("Training aborted by user"));
                            }
                        } else {
                            // Validation passed - load checkpoint and resume
                            if let Some(warning) = validation.data_check_warning() {
                                // stderr in both modes: an unchecked data file is
                                // worth a line even in headless output.
                                eprintln!("{warning}");
                            } else if !headless {
                                println!("✅ Checkpoint validated: model and data match");
                            }

                            // Warn about dtype mismatch (weights will be converted)
                            if !validation.dtype_matches && !headless {
                                eprintln!(
                                    "⚠️  dtype mismatch: checkpoint={}, current={}. Weights will be converted.",
                                    validation.checkpoint_dtype, validation.current_dtype
                                );
                            }

                            crate::model::load_checkpoint_into_model(&mut model, &checkpoint_path)?;
                            if !headless {
                                println!("Resuming from step {}", step);
                            }
                            (step, Some(meta_path), Some(checkpoint_path))
                        }
                    } else {
                        if !headless {
                            println!("⚠️  No data file specified, skipping validation");
                        }
                        crate::model::load_checkpoint_into_model(&mut model, &checkpoint_path)?;
                        if !headless {
                            println!("Resuming from step {}", step);
                        }
                        (step, Some(meta_path), Some(checkpoint_path))
                    }
                }
                None => {
                    // stderr, and NOT gated on `headless`. `--resume auto` is an
                    // explicit request to continue from existing weights; falling
                    // back to random init is the opposite of what was asked, and
                    // a from-scratch run is indistinguishable from a fine-tune in
                    // the JSON metrics until the loss refuses to drop. Under
                    // `--headless` this was silent, so a fine-tune that never
                    // loaded its pretrained weights looked like a working run.
                    eprintln!(
                        "⚠️  --resume auto found no checkpoint in '{}' — STARTING FROM RANDOM \
                         WEIGHTS. A checkpoint is only eligible if it is named \
                         `*_step_N.safetensors` AND has a matching \
                         `checkpoint_step_N.meta.json` beside it.",
                        config.trainer.checkpoint_dir
                    );
                    (0, None, None)
                }
            }
        } else {
            // Manual resume: load specified checkpoint
            if !headless {
                println!("Loading checkpoint from: {}", resume_arg);
            }
            crate::model::load_checkpoint_into_model(&mut model, resume_arg)?;
            if !headless {
                println!("Checkpoint loaded successfully!");
            }
            // `*_step_N` resumes at step N, `*_final` at the step its meta records.
            let (step, meta_path) = manual_resume_target(resume_arg)?;
            if !headless && step > 0 {
                println!("Resuming from step {}", step);
            }
            (step, meta_path, Some(resume_arg.clone()))
        }
    } else {
        (0, None, None)
    };
    progress.final_step = start_step;
    // Plan the optimizer restore before the long setup, so a missing file fails early.
    let optimizer_resume = plan_optimizer_resume(
        resume_weights.as_deref(),
        resume_meta_path.as_deref(),
        start_step,
    )?;

    if !headless {
        println!("Model initialized successfully");
    }

    // 4. Create Data Loader
    // Use batch_size and seq_len from config
    let batch_size = config.trainer.batch_size;
    let seq_len = config.model.max_seq_len;
    if !headless {
        println!(
            "Using batch_size={}, seq_len={} (from config)",
            batch_size, seq_len
        );
    }

    // Create base data loader
    let base_loader = if let Some(data_path) = &args.data_file {
        if !headless {
            println!(
                "Loading training data from memory-mapped file: {}",
                data_path
            );
        }
        LitDataLoader::<R>::from_file(data_path, batch_size, seq_len, device.clone())?
    } else {
        if !headless {
            println!("Creating dummy training data...");
        }
        let total_tokens = 1_000_000; // 1M tokens for demo
        let dummy_tokens = create_dummy_data(config.model.vocab_size, total_tokens);
        LitDataLoader::<R>::new(dummy_tokens, batch_size, seq_len, device.clone())
    };

    // Masked loss scoring is opt-in: without either `speech_layout` or
    // `target_mask` the loader emits no mask and every position is scored,
    // exactly as before. `try_with_speech_layout` rejects a corpus whose
    // layout sidecar disagrees with the configured layout, rather than
    // training on ids that mean something else. `LitConfig::validate` already
    // rejects a config that sets both, so the two arms below never overlap
    // on a config that reached this point through `LitConfig::from_yaml`.
    let base_loader = match (&config.model.speech_layout, &config.trainer.target_mask) {
        (Some(spec), None) => base_loader.try_with_speech_layout(spec.layout())?,
        (None, Some(spec)) => base_loader.with_target_mask(spec.target_mask()),
        (None, None) => base_loader,
        (Some(_), Some(_)) => {
            anyhow::bail!(
                "model.speech_layout and trainer.target_mask are mutually exclusive: both \
                 configure the loader's loss mask"
            );
        }
    };

    let row_order = if config.trainer.shuffle_rows {
        RowOrder::Shuffled {
            seed: config.trainer.seed.unwrap_or(0),
        }
    } else {
        RowOrder::Stream
    };
    let mut base_loader = base_loader.with_row_order(row_order);

    // Get dataset size before potentially wrapping in prefetch
    let dataset_size = base_loader.dataset_size();
    progress.dataset_size = Some(dataset_size);
    let row_layout = base_loader.row_layout()?;
    let start_position = resume_data_position(
        resume_meta_path.as_deref(),
        start_step,
        &row_layout,
        config.trainer.gradient_accumulation,
    )?;
    base_loader.seek(start_position)?;

    if !headless {
        let prefetch_msg = if args.prefetch > 0 {
            format!(" (prefetch={})", args.prefetch)
        } else {
            String::new()
        };
        println!(
            "Data loader ready with {} tokens in {} rows of {} (row order: {:?}){}",
            dataset_size,
            row_layout.num_rows(),
            seq_len + 1,
            row_order,
            prefetch_msg
        );
        println!(
            "Data position: epoch {} row {}\n",
            start_position.epoch, start_position.row
        );
    }

    // 5. Initialize Trainer
    let config_trainer = config.trainer.clone();

    // Whole batches per epoch; the partial batch at an epoch's end is dropped.
    let steps_per_epoch =
        row_layout.batches_per_epoch() as f64 / config_trainer.gradient_accumulation as f64;
    let total_steps = row_layout.total_steps(
        config_trainer.gradient_accumulation,
        config_trainer.num_epochs,
    );

    // Effective batch size is the number of samples per optimizer step
    let effective_batch_size = batch_size * config_trainer.gradient_accumulation;

    progress.config = Some(serde_json::json!({
        "model": {
            "name": config.model.name,
            "hidden_size": config.model.hidden_size,
            "num_layers": config.model.num_layers,
            "num_heads": config.model.num_heads,
            "num_experts": config.model.num_experts,
            "experts_per_tok": config.model.experts_per_tok,
            "vocab_size": config.model.vocab_size,
            "max_seq_len": config.model.max_seq_len,
            "dtype": config.model.dtype.to_string(),
        },
        "trainer": {
            "learning_rate": config.trainer.learning_rate,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "gradient_accumulation": config_trainer.gradient_accumulation,
            "max_steps": config.trainer.max_steps,
            "num_epochs": config.trainer.num_epochs,
            "effective_batch_size": effective_batch_size,
            "total_steps": total_steps,
        }
    }));

    // Stable TensorId parameter registration.
    // There is no global tape: the optimizer owns a HashMap keyed by the model's
    // logical TensorIds, and every step rebuilds Vars with those same IDs.
    let trainable = model.trainable_parameters();
    let parameter_ids: Vec<TensorId> = trainable.iter().map(|(id, _)| *id).collect();
    let mut params = parameter_tensor_map::<R>(trainable);
    let parameter_names: Vec<(String, TensorId)> = model
        .named_trainable_parameters()?
        .into_iter()
        .map(|(name, id, _)| (name, id))
        .collect();

    let trainable_params = params
        .values()
        .map(|tensor| tensor.elem_count())
        .sum::<usize>();
    let total_params = trainable_params;

    let steps_json_path = Path::new(run_dir)
        .join(STEPS_FILE)
        .to_string_lossy()
        .into_owned();
    let mut trainer = LitTrainer::<R>::new(
        config_trainer.clone(),
        &mut params,
        parameter_ids,
        parameter_names,
        client.clone(),
        device.clone(),
        TrainerRunConfig {
            row_layout,
            start_position,
            batch_size,
            steps_json_path: steps_json_path.clone(),
            headless,
            start_step,
            model_config: config.model.clone(),
            data_file_path: args.data_file.clone(),
        },
    )?;
    trainer.apply_optimizer_resume(&optimizer_resume)?;

    if !headless {
        println!("Trainer initialized with AdamW optimizer\n");
    }

    // Show available GPU memory (only in normal mode)
    if !headless {
        if let Ok(mem_info) = get_cuda_memory_info::<R>(&device) {
            println!("GPU Memory: {}\n", mem_info);
        }
    }

    // Print training banner (only in normal mode)
    if !headless {
        println!("\n============================================================");
        println!("🦥 OXIDIZR TRAINING");
        println!("============================================================\n");

        println!("📋 Configuration:");
        let param_str = estimate_params(&config.model);
        let num_mamba = config
            .model
            .mamba_layers
            .as_ref()
            .map(|v| v.len())
            .unwrap_or(0);
        let num_attention = config.model.num_layers - num_mamba;
        let mamba_type = if config.model.mamba2_num_heads.is_some() {
            "Mamba2"
        } else {
            "Mamba"
        };
        println!(
            "  Model: {} parameters ({} {} + {} MLA+MoE layers)",
            param_str, num_mamba, mamba_type, num_attention
        );
        println!("  Dataset: {} tokens", dataset_size);
        println!("  Checkpoint Dir: {}", config_trainer.checkpoint_dir);
        println!();
        println!("  Max Seq Length: {}", config.model.max_seq_len);
        if let Some(num_experts) = config.model.num_experts {
            println!(
                "  MoE: {} experts, Top-{} routing, Load balance α={:.3}",
                num_experts,
                config.model.experts_per_tok.unwrap_or(1),
                config_trainer.load_balance_alpha
            );
        }
        println!(
            "  Batch Size: {}, Gradient Accumulation: {}",
            batch_size, config_trainer.gradient_accumulation
        );
        println!("  Learning Rate: {:.6}", config_trainer.learning_rate);
        println!("  Epochs: {}", config_trainer.num_epochs);
        println!();

        println!("📊 Training Plan:");
        println!(
            "  Trainable params: {} || all params: {} || trainable%: {:.2}",
            trainable_params,
            total_params,
            (trainable_params as f64 / total_params as f64) * 100.0
        );
        println!(
            "  Rows = {} | Steps per epoch = {:.2} | Num Epochs = {} | Total steps = {}",
            row_layout.num_rows(),
            steps_per_epoch,
            config_trainer.num_epochs,
            total_steps
        );
        println!(
            "  Batch size per device = {} | Gradient accumulation = {}",
            batch_size, config_trainer.gradient_accumulation
        );
        let actual_batch = batch_size * config_trainer.gradient_accumulation;
        println!(
            "  Total batch size ({} x {}) = {}",
            batch_size, config_trainer.gradient_accumulation, actual_batch
        );
        println!();

        println!("============================================================");
        println!("🚀 STARTING TRAINING");
        println!("============================================================");
        println!();
        println!("Training in progress... (progress bar should appear below)");
        println!("If no progress bar appears, use --headless for text output.\n");
    }

    // 6. Run Training Loop with better OOM error handling
    // Use prefetch loader if enabled, otherwise use base loader
    let training_result = if args.prefetch > 0 {
        let mut prefetch_loader = PrefetchDataLoader::<R>::new(base_loader, args.prefetch)?;
        trainer.fit(&model, &mut prefetch_loader)
    } else {
        let mut data_loader = base_loader;
        trainer.fit(&model, &mut data_loader)
    };
    progress.final_step = trainer.global_step();
    progress.last_loss = trainer.last_loss();

    match training_result {
        Ok(_) => {
            if !headless {
                println!(
                    "\nTraining complete! Checkpoints saved to: {}",
                    config.trainer.checkpoint_dir
                );
            }
            Ok(())
        }
        Err(e) => {
            // Check if it's an OOM error
            let err_str = format!("{:?}", e);
            if err_str.contains("out of memory") || err_str.contains("OUT_OF_MEMORY") {
                if !headless {
                    println!("\n❌ CUDA OUT OF MEMORY ERROR ❌");
                    if let Ok(mem_info) = get_cuda_memory_info::<R>(&device) {
                        println!("GPU Memory at failure: {}", mem_info);
                    }
                    println!("\nCurrent configuration:");
                    println!("  - Model: {} parameters", estimate_params(&config.model));
                    println!("  - Batch size: {}", batch_size);
                    println!("  - Sequence length: {}", seq_len);
                    println!("  - Hidden size: {}", config.model.hidden_size);
                    println!("  - Number of layers: {}", config.model.num_layers);
                    println!(
                        "  - Number of experts: {}",
                        config.model.num_experts.unwrap_or(0)
                    );
                    println!("\nSuggestions:");
                    println!("  1. Reduce batch_size (currently {})", batch_size);
                    println!("  2. Reduce seq_len (currently {})", seq_len);
                    println!("  3. Reduce model size (hidden_size, num_layers, or num_experts)");
                    println!("  4. Use --dtype bf16 or --dtype f16 for lower memory usage");
                    println!();
                }
                // Keep the device error under the summary, so run.json records both.
                Err(anyhow::Error::from(e).context("CUDA out of memory"))
            } else {
                Err(e.into())
            }
        }
    }
}
