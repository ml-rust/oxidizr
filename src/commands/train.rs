//! Training command implementation
//!
//! This module contains the training logic, extracted from the original main.rs.

use anyhow::Result;
use crate::prelude::{Device, VarBuilder};
use clap::Args;

use crate::config::{find_latest_checkpoint, CheckpointMeta, DTypePrecision, LitConfig, ModelConfig};
use crate::data::{create_dummy_data, LitDataLoader, PrefetchDataLoader};
use crate::model::LitGPT;
use crate::trainer::LitTrainer;

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

    /// Gradient synchronization backend: "single", "cpu", or "nccl"
    #[arg(long = "sync-backend", default_value = "cpu")]
    pub sync_backend: String,

    /// Enable async data prefetching (CPU-GPU overlap)
    /// Prefetches N batches in background thread while GPU trains
    #[arg(long = "prefetch", default_value = "0")]
    pub prefetch: usize,

    /// Maximum checkpoints to keep (0 = unlimited, default = 10)
    #[arg(long = "max-checkpoints")]
    pub max_checkpoints: Option<usize>,

    /// Model precision: f32 (default), f16, or bf16
    /// BF16 recommended for tensor core acceleration on modern GPUs
    #[arg(long = "dtype", default_value = "f32")]
    pub dtype: String,
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
    io::stdout().flush().unwrap();

    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_err() {
        return false;
    }

    matches!(input.trim().to_lowercase().as_str(), "y" | "yes")
}

/// Clear all checkpoint files from a directory
/// Removes .safetensors, .meta.json, and config.json files
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
            // Remove checkpoint files (.safetensors), metadata (.meta.json), and config.json
            if filename.ends_with(".safetensors")
                || filename.ends_with(".meta.json")
                || filename == "config.json"
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

/// Get CUDA memory info in a human-readable format
fn get_cuda_memory_info(_device: &Device) -> Result<String> {
    // Just query all GPUs - simpler and more reliable
    let output = std::process::Command::new("nvidia-smi")
        .args(&[
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
pub fn estimate_params(cfg: &ModelConfig) -> String {
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

    // LM Head: hidden_size * vocab_size
    let lm_head_params = cfg.hidden_size * cfg.vocab_size;

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

/// Run the training command
pub fn run_train(args: TrainArgs) -> Result<()> {
    // Use run directory from env var (set by wrapper script) or create new one
    let run_dir = match std::env::var("OXIDIZED_RUN_DIR") {
        Ok(dir) => {
            // Wrapper script already created the directory
            dir
        }
        Err(_) => {
            // No wrapper - create our own timestamped directory
            let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
            let dir = format!("./runs/{}", timestamp);
            std::fs::create_dir_all(&dir)?;
            dir
        }
    };

    // Apply headless mode flag
    let headless = args.headless;

    if !headless {
        println!("=== Oxidizr Training Framework ===");
        println!("Run directory: {}\n", run_dir);
    }

    // 1. Load Configuration from YAML file
    if !headless {
        println!("Loading configuration from: {}", args.config_file);
    }
    let mut config = LitConfig::from_yaml(&args.config_file)?;

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
        config.trainer.distributed.backend = args.sync_backend.parse().unwrap_or_default();
        if !headless {
            println!(
                "CLI override: distributed training enabled with {} GPUs: {:?}",
                gpu_ids.len(), gpu_ids
            );
            println!(
                "CLI override: sync_backend = {:?}",
                config.trainer.distributed.backend
            );
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

    // Apply dtype override
    if let Some(dtype) = DTypePrecision::from_str(&args.dtype) {
        if dtype != config.model.dtype {
            if !headless {
                println!(
                    "CLI override: dtype = {} (was {})",
                    dtype, config.model.dtype
                );
            }
            config.model.dtype = dtype;
        }
    } else {
        eprintln!("Warning: Unknown dtype '{}', using config default ({})",
                  args.dtype, config.model.dtype);
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

    // 2. Setup Device (priority: CLI override > YAML config > default GPU)
    let use_cpu = if let Some(ref dev) = args.target_device {
        dev.to_lowercase() == "cpu"
    } else {
        config.trainer.target_device == crate::config::TargetDevice::Cpu
    };

    let device = if use_cpu {
        if !headless {
            println!("Using CPU device");
        }
        Device::cpu()
    } else {
        Device::cuda_if_available()?
    };
    if !headless {
        println!("Using device: {:?}", device);
    }

    // 3. Initialize Model
    if !headless {
        println!(
            "Initializing model with {} parameters...",
            estimate_params(&config.model)
        );
    }

    // For training, we use VarBuilder with PyTorch's default Linear initialization
    // PyTorch uses Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) which produces smaller
    // initial values than Kaiming, helping with gradient stability in Mamba layers.
    // Use configured dtype (F32/F16/BF16) for model weights
    // RmsNorm and SSM computations use F32 internally for numerical stability
    let vb = VarBuilder::pytorch_linear(config.model.dtype.to_dtype(), &device);
    let model = LitGPT::new(&config.model, vb.clone())?;

    // Get the VarMap from VarBuilder (contains all initialized weights)
    let varmap_arc = vb.var_map();

    // Load checkpoint if resuming
    let start_step = if let Some(ref resume_arg) = args.resume {
        if resume_arg == "auto" {
            // Auto-resume: find latest checkpoint and validate
            if !headless {
                println!("Auto-resume: searching for latest checkpoint in {}", config.trainer.checkpoint_dir);
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
                            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                            if !validation.model_matches {
                                println!("❌ Model config mismatch:");
                                println!("   Checkpoint hash: {}", validation.checkpoint_model_hash);
                                println!("   Current hash:    {}", validation.current_model_hash);
                            }
                            if !validation.data_matches {
                                println!("❌ Data file mismatch:");
                                println!("   Checkpoint hash: {}", validation.checkpoint_data_hash);
                                println!("   Current hash:    {}", validation.current_data_hash);
                            }
                            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                            println!("\nThe current configuration does not match the existing checkpoint.");
                            println!("Starting fresh training will REMOVE all existing checkpoints in:");
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
                            if prompt_confirmation("Do you want to clear old checkpoints and start fresh?") {
                                // Clear old checkpoints
                                match clear_checkpoint_directory(&config.trainer.checkpoint_dir) {
                                    Ok(count) => {
                                        println!("🗑️  Removed {} checkpoint file(s)", count);
                                        println!("Starting fresh training from step 0\n");
                                    }
                                    Err(e) => {
                                        eprintln!("Error clearing checkpoints: {}", e);
                                        return Err(anyhow::anyhow!("Failed to clear checkpoint directory"));
                                    }
                                }
                                0 // Start from step 0
                            } else {
                                println!("Aborted. Please resolve the configuration mismatch or use a different checkpoint directory.");
                                return Err(anyhow::anyhow!("Training aborted by user"));
                            }
                        } else {
                            // Validation passed - load checkpoint and resume
                            if !headless {
                                println!("✅ Checkpoint validated: model and data match");
                            }

                            // Warn about dtype mismatch (weights will be converted)
                            if !validation.dtype_matches && !headless {
                                eprintln!(
                                    "⚠️  dtype mismatch: checkpoint={}, current={}. Weights will be converted.",
                                    validation.checkpoint_dtype, validation.current_dtype
                                );
                            }

                            varmap_arc.write().unwrap().load(&checkpoint_path)?;
                            if !headless {
                                println!("Resuming from step {}", step);
                            }
                            step
                        }
                    } else {
                        if !headless {
                            println!("⚠️  No data file specified, skipping validation");
                        }
                        varmap_arc.write().unwrap().load(&checkpoint_path)?;
                        if !headless {
                            println!("Resuming from step {}", step);
                        }
                        step
                    }
                }
                None => {
                    if !headless {
                        println!("No checkpoint found, starting fresh");
                    }
                    0
                }
            }
        } else {
            // Manual resume: load specified checkpoint
            if !headless {
                println!("Loading checkpoint from: {}", resume_arg);
            }
            varmap_arc.write().unwrap().load(resume_arg)?;
            if !headless {
                println!("Checkpoint loaded successfully!");
            }
            // Try to parse step number from checkpoint filename
            // Format: {name}_{seq}_{params}_{dtype}_step_{N} (e.g., nano_512_117m_bf16_step_100)
            let step = std::path::Path::new(resume_arg)
                .file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| {
                    if s.contains("_step_") {
                        s.split("_step_").last()
                            .and_then(|n| n.parse::<usize>().ok())
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            if !headless && step > 0 {
                println!("Resuming from step {}", step);
            }
            step
        }
    } else {
        0
    };

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
        LitDataLoader::from_file(data_path, batch_size, seq_len, device.clone())?
    } else {
        if !headless {
            println!("Creating dummy training data...");
        }
        let total_tokens = 1_000_000; // 1M tokens for demo
        let dummy_tokens = create_dummy_data(config.model.vocab_size, total_tokens);
        LitDataLoader::new(dummy_tokens, batch_size, seq_len, device.clone())
    };

    // Get dataset size before potentially wrapping in prefetch
    let dataset_size = base_loader.dataset_size();

    if !headless {
        let prefetch_msg = if args.prefetch > 0 {
            format!(" (prefetch={})", args.prefetch)
        } else {
            String::new()
        };
        println!(
            "Data loader ready with {} tokens{}\n",
            dataset_size, prefetch_msg
        );
    }

    // 5. Initialize Trainer
    let config_trainer = config.trainer.clone();

    // For step calculation, we need total tokens per step
    let tokens_per_step = batch_size * seq_len * config_trainer.gradient_accumulation;
    let steps_per_epoch = (dataset_size + tokens_per_step - 1) / tokens_per_step;
    let total_steps = steps_per_epoch * config_trainer.num_epochs;

    // Effective batch size is the number of samples per optimizer step
    let effective_batch_size = batch_size * config_trainer.gradient_accumulation;

    // Extract VarMap from the Arc<RwLock> for use with the trainer
    // At this point we're done with VarBuilder, so we can clone out the VarMap
    let mut varmap = varmap_arc.read().unwrap().clone();

    // Calculate total parameters and mark them as requiring gradients
    let (total_params, trainable_params) = {
        use crate::prelude::global_tape;

        let mut total = 0usize;
        for tensor in varmap.all_vars() {
            total += tensor.elem_count();
            // Mark this tensor as requiring gradients and record it in the tape
            global_tape::set_requires_grad(tensor.id(), true);
            global_tape::record_tensor(tensor.id(), tensor.clone());
        }
        (total, total) // For now, all params are trainable
    };

    let steps_json_path = format!("{}/steps.json", run_dir);
    let mut trainer = LitTrainer::new(
        config_trainer.clone(),
        &mut varmap,
        device.clone(),
        dataset_size,
        batch_size,
        seq_len,
        steps_json_path.clone(),
        headless,
        start_step,
        config.model.clone(),
        args.data_file.clone(),
    )?;

    if !headless {
        println!("Trainer initialized with AdamW optimizer\n");
    }

    // Show available GPU memory (only in normal mode)
    if !headless {
        if let Ok(mem_info) = get_cuda_memory_info(&device) {
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
        let mamba_type = if config.model.mamba2_num_heads.is_some() { "Mamba2" } else { "Mamba" };
        println!(
            "  Model: {} parameters ({} {} + {} MLA+MoE layers)",
            param_str, num_mamba, mamba_type, num_attention
        );
        println!("  Dataset: {} tokens", dataset_size);
        println!("  Checkpoint Dir: {}", config_trainer.checkpoint_dir);
        println!("");
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
        println!("");

        println!("📊 Training Plan:");
        println!(
            "  Trainable params: {} || all params: {} || trainable%: {:.2}",
            trainable_params,
            total_params,
            (trainable_params as f64 / total_params as f64) * 100.0
        );
        println!(
            "  Num examples = {} | Num Epochs = {} | Total steps = {}",
            steps_per_epoch, config_trainer.num_epochs, total_steps
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
        println!("");

        println!("============================================================");
        println!("🚀 STARTING TRAINING");
        println!("============================================================");
        println!("");
        println!("Training in progress... (progress bar should appear below)");
        println!("If no progress bar appears, use --headless for text output.\n");
    }

    // 6. Run Training Loop with better OOM error handling
    // Use prefetch loader if enabled, otherwise use base loader
    let training_result = if args.prefetch > 0 {
        let mut prefetch_loader = PrefetchDataLoader::new(base_loader, args.prefetch)?;
        trainer.fit(&model, &mut prefetch_loader)
    } else {
        let mut data_loader = base_loader;
        trainer.fit(&model, &mut data_loader)
    };

    // Write run.json with metadata
    let run_json = serde_json::json!({
        "run_dir": run_dir,
        "config": {
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
        },
        "dataset_size": dataset_size,
        "device": format!("{:?}", device),
        "status": if training_result.is_ok() { "completed" } else { "failed" },
        "error": if let Err(ref e) = training_result {
            Some(format!("{:?}", e))
        } else {
            None
        },
    });

    let run_json_path = format!("{}/run.json", run_dir);
    if let Ok(run_json_str) = serde_json::to_string_pretty(&run_json) {
        let _ = std::fs::write(&run_json_path, run_json_str);
    }

    match training_result {
        Ok(_) => {
            if !headless {
                println!(
                    "\nTraining complete! Checkpoints saved to: {}",
                    config.trainer.checkpoint_dir
                );
                println!("Run metadata saved to: {}/run.json", run_dir);
            }
            Ok(())
        }
        Err(e) => {
            // Check if it's an OOM error
            let err_str = format!("{:?}", e);
            if err_str.contains("out of memory") || err_str.contains("OUT_OF_MEMORY") {
                if !headless {
                    println!("\n❌ CUDA OUT OF MEMORY ERROR ❌");
                    if let Ok(mem_info) = get_cuda_memory_info(&device) {
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
                    println!("");
                }
                Err(anyhow::anyhow!("CUDA out of memory"))
            } else {
                Err(e.into())
            }
        }
    }
}
