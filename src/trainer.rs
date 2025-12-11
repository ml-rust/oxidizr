use crate::config::{BlazrConfig, CheckpointMeta, ModelConfig, TrainerConfig};
use crate::data::DataLoader;
use crate::distributed::{
    compute_grad_norm, create_synchronizer, AdamWConfig as DistAdamWConfig, DistributedAdamW,
    GradientSynchronizer,
};
use crate::prelude::{cross_entropy, Device, Error, FusedAdamW, GradStore, Result, Tensor, VarMap};
use boostr::AdamWConfig;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::VecDeque;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

/// The "LitGPT" style Trainer abstraction.
/// It owns the loop state, the optimizer, and handles the "boring" stuff.
pub struct LitTrainer<'a> {
    config: TrainerConfig,
    varmap: &'a mut VarMap,
    optimizer: FusedAdamW,
    #[allow(dead_code)] // Kept for potential future use (e.g., device-specific optimizations)
    device: Device,
    global_step: usize,
    // Training metrics
    dataset_size: usize,
    total_steps: usize,
    batch_size: usize,
    seq_len: usize,
    // JSON logging
    steps_json_path: String,
    // UI mode
    headless: bool,
    // Gradient clipping state (for loss pre-scaling heuristic)
    prev_grad_norm: Option<f64>,
    // Resume support
    start_step: usize,
    // Checkpoint metadata
    model_config: ModelConfig,
    data_file_path: Option<String>,
    // Distributed training support
    synchronizer: Option<Box<dyn GradientSynchronizer>>,
    distributed_optimizer: Option<DistributedAdamW>,
    // Learning rate warmup (critical for Mamba SSM parameter alignment)
    warmup_steps: usize,     // Effective warmup steps (0 = disabled)
    base_learning_rate: f64, // Base LR to warmup towards
}

impl<'a> LitTrainer<'a> {
    pub fn new(
        config: TrainerConfig,
        varmap: &'a mut VarMap,
        device: Device,
        dataset_size: usize,
        batch_size: usize,
        seq_len: usize,
        steps_json_path: String,
        headless: bool,
        start_step: usize,
        model_config: ModelConfig,
        data_file_path: Option<String>,
    ) -> Result<Self> {
        // Init FusedAdamW with configurable SOTA parameters (from optimization.md)
        // - beta2 = 0.95: Lower β₂ stabilizes MoE sparse expert updates
        // - eps = 1e-6: Higher ε for BF16 numerical safety
        // - weight_decay = 0.1: Industry standard (decoupled from LR)
        let optimizer = FusedAdamW::from_varmap_with_config(
            varmap,
            AdamWConfig {
                lr: config.learning_rate as f32,
                beta1: config.adamw_beta1 as f32,
                beta2: config.adamw_beta2 as f32,
                eps: config.adamw_eps as f32,
                weight_decay: config.weight_decay as f32,
            },
        );

        // Calculate total steps from dataset
        let effective_batch_size = batch_size * seq_len * config.gradient_accumulation;
        let steps_per_epoch = (dataset_size + effective_batch_size - 1) / effective_batch_size; // Ceiling division
        let total_steps = steps_per_epoch * config.num_epochs;

        // Initialize distributed training components if enabled
        let (synchronizer, distributed_optimizer) = if config.distributed.enabled {
            let sync = create_synchronizer(&config.distributed)?;
            let dist_opt =
                DistributedAdamW::new(varmap, DistAdamWConfig::with_lr(config.learning_rate))?;
            (Some(sync), Some(dist_opt))
        } else {
            (None, None)
        };

        // Calculate effective warmup steps (explicit steps OR ratio * total_steps)
        // Priority: warmup_steps > warmup_ratio > disabled (0)
        let warmup_steps = if let Some(steps) = config.warmup_steps {
            steps
        } else if let Some(ratio) = config.warmup_ratio {
            (total_steps as f64 * ratio).ceil() as usize
        } else {
            0 // No warmup
        };

        // Store base learning rate for warmup calculation
        let base_learning_rate = config.learning_rate;

        Ok(Self {
            config,
            varmap,
            optimizer,
            device,
            global_step: start_step,
            dataset_size,
            total_steps,
            batch_size,
            seq_len,
            steps_json_path,
            headless,
            prev_grad_norm: None,
            start_step,
            model_config,
            data_file_path,
            synchronizer,
            distributed_optimizer,
            warmup_steps,
            base_learning_rate,
        })
    }

    /// Performs one training step (including gradient accumulation)
    /// Returns (loss, grad_norm)
    ///
    /// # Gradient Clipping Strategy
    /// Since Candle 0.9.x doesn't support in-place gradient modification,
    /// we use loss pre-scaling based on the previous step's gradient norm.
    /// This effectively clips gradients while preserving AdamW's momentum state.
    pub fn training_step<D: DataLoader>(
        &mut self,
        model: &crate::model::LitGPT,
        data_loader: &mut D,
    ) -> Result<(f64, f64)> {
        // Apply learning rate warmup at the START of each step (critical for Mamba SSM alignment)
        // This ensures consistent LR throughout forward/backward/optimizer
        if self.warmup_steps > 0 && self.global_step < self.warmup_steps {
            let warmup_ratio = (self.global_step + 1) as f64 / self.warmup_steps as f64;
            let warmup_lr = (self.base_learning_rate * warmup_ratio).max(1e-7);
            self.optimizer.set_lr(warmup_lr as f32);
        } else if self.warmup_steps > 0 && self.global_step == self.warmup_steps {
            // Restore full learning rate after warmup completes
            self.optimizer.set_lr(self.base_learning_rate as f32);
        }

        let mut total_loss_for_logging = 0.0;
        #[allow(unused_assignments)]
        let mut last_grad_norm = 0.0;
        let accum_steps = self.config.gradient_accumulation;
        let load_balance_alpha = self.config.load_balance_alpha;

        // Profiling support - set PROFILE_TRAIN=1 to enable detailed timing
        let profile = std::env::var("PROFILE_TRAIN").is_ok();
        let step_start = if profile { Some(Instant::now()) } else { None };

        // Note: max_grad_value (per-element clipping) is handled by FusedAdamW's built-in clamping (±1e4)
        // Global gradient clipping and normalization are applied below before optimizer.step()
        let _max_grad_value = self.config.max_grad_value;

        // Accumulated loss tensor across micro-batches
        // We accumulate losses as tensors (keeping computation graph), then call backward once
        let mut accumulated_loss: Option<Tensor> = None;

        // Gradient Accumulation Loop - only forward passes, accumulate losses
        for _micro_batch_idx in 0..accum_steps {
            let data_start = if profile { Some(Instant::now()) } else { None };
            let (input, target) = data_loader.next_batch()?;
            if let Some(start) = data_start {
                eprintln!("[PROFILE] data_load: {:.3}ms", start.elapsed().as_secs_f64() * 1000.0);
            }

            // Register parameters as requires_grad on the tape for this forward pass
            // This ensures gradient computation works correctly after tape.clear()
            let reg_start = if profile { Some(Instant::now()) } else { None };
            self.varmap.register_requires_grad();
            if let Some(start) = reg_start {
                eprintln!("[PROFILE] register_requires_grad: {:.3}ms", start.elapsed().as_secs_f64() * 1000.0);
            }

            // Forward pass (returns logits + auxiliary losses + z-losses)
            // Wrap in catch_unwind to provide user-friendly OOM error
            let fwd_start = if profile { Some(Instant::now()) } else { None };
            let output = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                model.forward(&input)
            })) {
                Ok(result) => result?,
                Err(panic_info) => {
                    let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_info.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "Unknown panic".to_string()
                    };

                    if panic_msg.contains("can't allocate buffer")
                        || panic_msg.contains("out of memory")
                    {
                        let size_info = if let Some(start) = panic_msg.find("size:") {
                            let size_str = &panic_msg[start + 5..];
                            if let Some(size) = size_str.trim().split_whitespace().next() {
                                if let Ok(bytes) = size.parse::<u64>() {
                                    format!(" ({:.1} MB)", bytes as f64 / 1_000_000.0)
                                } else {
                                    String::new()
                                }
                            } else {
                                String::new()
                            }
                        } else {
                            String::new()
                        };

                        return Err(Error::Msg(format!(
                            "GPU out of memory during forward pass{}\n\n\
                            The model requires more VRAM than available. Try:\n\
                            • Reduce batch_size (current: {})\n\
                            • Reduce max_seq_len in config\n\
                            • Increase gradient_accumulation to maintain effective batch size\n\
                            • Use a smaller model configuration\n\
                            • Free GPU memory from other processes (check with: nvidia-smi)",
                            size_info, self.batch_size
                        )));
                    }

                    std::panic::resume_unwind(panic_info);
                }
            };
            if let Some(start) = fwd_start {
                eprintln!("[PROFILE] forward: {:.3}ms", start.elapsed().as_secs_f64() * 1000.0);
            }
            let logits = output.logits;
            let aux_losses = output.aux_losses;
            let z_losses = output.z_losses;

            // Check for NaN in logits (indicates model forward pass issue)
            // EXPENSIVE: This transfers entire logits tensor to CPU - only do on first step or if DEBUG
            let nan_check_start = if profile { Some(Instant::now()) } else { None };
            if self.global_step == 0 || std::env::var("DEBUG_NAN").is_ok() {
                if let Ok(logits_vec) = logits.flatten_all()?.to_vec::<f32>() {
                    if logits_vec.iter().any(|x| x.is_nan() || x.is_infinite()) {
                        eprintln!("\n❌ NaN/Inf in model logits! This indicates:");
                        eprintln!("   1. Numerical instability in forward pass");
                        eprintln!("   2. RoPE or attention computation overflow");
                        eprintln!("   3. Mamba SSM state explosion");
                        return Err(Error::Msg("NaN in logits".to_string()));
                    }
                }
            }
            if let Some(start) = nan_check_start {
                eprintln!("[PROFILE] nan_check: {:.3}ms", start.elapsed().as_secs_f64() * 1000.0);
            }

            // Calculate Loss (Cross Entropy)
            // Flatten batch/seq dims for cross_entropy: [Batch*Seq, Vocab]
            // Note: logits are already F32 from model forward (cast for stable loss computation)
            // Use reshape_tracked to record on tape for gradient flow
            let loss_start = if profile { Some(Instant::now()) } else { None };
            let (b, s, v) = logits.dims3()?;
            let logits_flat = logits.reshape_tracked((b * s, v))?;
            let target_flat = target.reshape(b * s)?;

            let ce_loss_per_token = cross_entropy(&logits_flat, &target_flat)?;
            let ce_loss = ce_loss_per_token.mean_all()?;
            // Note: mean_all() returns a scalar on CPU but may need squeezing on CUDA in some cases

            // NaN checking is expensive (GPU->CPU sync) - only do on first step or when DEBUG_NAN is set
            let check_nan = self.global_step == 0 || std::env::var("DEBUG_NAN").is_ok();

            // Check for NaN in CE loss before combining (only when check_nan is true)
            if check_nan {
                let ce_loss_val = ce_loss.to_vec0::<f32>()? as f64;
                if ce_loss_val.is_nan() || ce_loss_val.is_infinite() {
                    eprintln!("\n❌ NaN/Inf detected in CE loss! This usually indicates:");
                    eprintln!("   1. Learning rate too high (try 1e-4 or lower)");
                    eprintln!("   2. Gradient explosion (sequence too long)");
                    eprintln!("   3. Numerical instability in model");
                    eprintln!("   Current CE loss: {}", ce_loss_val);
                    return Err(Error::Msg("NaN loss detected".to_string()));
                }
            }

            // Add auxiliary losses (e.g., MoE load balancing) - keep as tensors for gradient flow
            // IMPORTANT: Don't use to_vec0 here as it breaks the gradient chain!
            let mut loss = ce_loss.clone();
            if !aux_losses.is_empty() {
                // Sum all aux losses as tensors, then scale by alpha/count
                let num_aux = aux_losses.len() as f32;
                for aux_loss in &aux_losses {
                    // CRITICAL: CUDA can return [1,1] instead of scalar - squeeze manually
                    let mut loss_t = aux_loss.clone();
                    while loss_t.rank() > 0 && loss_t.elem_count() == 1 {
                        loss_t = loss_t.squeeze(0)?;
                    }
                    // Add scaled aux loss to total loss (preserves gradient chain)
                    let scaled_aux = loss_t.scale((load_balance_alpha / num_aux as f64) as f32)?;
                    loss = loss.add(&scaled_aux)?;
                }
            }

            // Add router z-losses (prevents MoE logit explosion in BF16)
            // L_total = L_MLE + α*L_aux + γ*L_z where γ = z_loss_alpha
            let z_loss_alpha = self.config.z_loss_alpha;
            if !z_losses.is_empty() && z_loss_alpha > 0.0 {
                let num_z = z_losses.len() as f32;
                for z_loss in &z_losses {
                    // CRITICAL: CUDA can return [1,1] instead of scalar - squeeze manually
                    let mut loss_t = z_loss.clone();
                    while loss_t.rank() > 0 && loss_t.elem_count() == 1 {
                        loss_t = loss_t.squeeze(0)?;
                    }
                    // Add scaled z-loss to total loss (preserves gradient chain)
                    let scaled_z = loss_t.scale((z_loss_alpha / num_z as f64) as f32)?;
                    loss = loss.add(&scaled_z)?;
                }
            }

            // Check combined loss for NaN (only when check_nan is true)
            if check_nan {
                let total_loss_val = loss.to_vec0::<f32>()? as f64;
                if total_loss_val.is_nan() || total_loss_val.is_infinite() {
                    eprintln!("\n❌ NaN/Inf in combined loss!");
                    return Err(Error::Msg("NaN loss detected".to_string()));
                }
            }

            // Scale loss for gradient accumulation
            let scaled_loss = loss.clone().mul_scalar((1.0 / accum_steps as f64) as f32)?;

            // Accumulate scaled loss (keeps computation graph for single backward)
            accumulated_loss = Some(match accumulated_loss {
                None => scaled_loss,
                Some(acc) => acc.add(&scaled_loss)?,
            });

            // Track loss tensor for later logging (defer GPU->CPU transfer)
            // We'll do the transfer after backward when we need to log anyway
            if let Some(start) = loss_start {
                eprintln!("[PROFILE] loss_compute: {:.3}ms", start.elapsed().as_secs_f64() * 1000.0);
            }
        }

        // Transfer loss to CPU for logging (single sync point)
        let loss_sync_start = if profile { Some(Instant::now()) } else { None };
        if let Some(ref acc_loss) = accumulated_loss {
            // Transfer the accumulated loss value
            total_loss_for_logging = acc_loss.to_vec0::<f32>()? as f64 * accum_steps as f64;
        }
        if let Some(start) = loss_sync_start {
            eprintln!("[PROFILE] loss_sync: {:.3}ms", start.elapsed().as_secs_f64() * 1000.0);
        }

        // ===== AFTER ACCUMULATION LOOP: Single backward pass =====
        // All micro-batches have been processed, now do backward on accumulated loss

        let total_loss =
            accumulated_loss.ok_or_else(|| Error::Msg("No loss accumulated".to_string()))?;

        // Single backward pass over all micro-batches
        // Wrap in catch_unwind to provide user-friendly OOM error instead of panic
        let bwd_start = if profile { Some(Instant::now()) } else { None };
        let grads = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            total_loss.backward()
        })) {
            Ok(result) => result?,
            Err(panic_info) => {
                // Extract panic message
                let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic".to_string()
                };

                // Check if it's an OOM error
                if panic_msg.contains("can't allocate buffer")
                    || panic_msg.contains("out of memory")
                {
                    // Extract the requested size if present
                    let size_info = if let Some(start) = panic_msg.find("size:") {
                        let size_str = &panic_msg[start + 5..];
                        if let Some(size) = size_str.trim().split_whitespace().next() {
                            if let Ok(bytes) = size.parse::<u64>() {
                                format!(" ({:.1} MB)", bytes as f64 / 1_000_000.0)
                            } else {
                                String::new()
                            }
                        } else {
                            String::new()
                        }
                    } else {
                        String::new()
                    };

                    return Err(Error::Msg(format!(
                        "GPU out of memory during backward pass{}\n\n\
                        The model requires more VRAM than available. Try:\n\
                        • Reduce batch_size (current: {})\n\
                        • Reduce max_seq_len in config\n\
                        • Increase gradient_accumulation to maintain effective batch size\n\
                        • Use a smaller model configuration\n\
                        • Free GPU memory from other processes (check with: nvidia-smi)",
                        size_info, self.batch_size
                    )));
                }

                // Re-panic for other errors
                std::panic::resume_unwind(panic_info);
            }
        };
        if let Some(start) = bwd_start {
            eprintln!("[PROFILE] backward: {:.3}ms", start.elapsed().as_secs_f64() * 1000.0);
        }

        // Check parameter/gradient matching at step 1
        if self.global_step == 0 {
            let total_params = self.varmap.all_vars_named().count();
            let grads_count = grads.len();
            let mut params_with_grads = 0;
            let mut params_without_grads = Vec::new();
            for (name, tensor) in self.varmap.all_vars_named() {
                if grads.get(tensor.id()).is_some() {
                    params_with_grads += 1;
                } else {
                    params_without_grads.push(name.clone());
                }
            }
            eprintln!(
                "[GRAD CHECK] Total params: {}, Grads returned: {}, Params with grads: {}",
                total_params, grads_count, params_with_grads
            );
            if !params_without_grads.is_empty() {
                eprintln!(
                    "[GRAD CHECK] Params WITHOUT gradients ({}):",
                    params_without_grads.len()
                );
                for name in &params_without_grads {
                    eprintln!("  - {}", name);
                }
            }
        }

        // Apply gradients - either single-GPU or distributed path
        if let (Some(ref sync), Some(ref mut dist_opt)) =
            (&self.synchronizer, &mut self.distributed_optimizer)
        {
            // DISTRIBUTED PATH: Synchronize gradients across GPUs
            // 1. Synchronize gradients (all-reduce across GPUs)
            let synced_grads = sync.synchronize_gradients(&grads, self.varmap)?;

            // 2. Compute gradient norm from synchronized gradients
            let grad_norm = compute_grad_norm(&synced_grads)?;
            last_grad_norm = grad_norm;

            // 3. Apply gradients via distributed optimizer
            dist_opt.step(&synced_grads, self.varmap)?;
        } else {
            // SINGLE-GPU PATH
            // 1. Compute gradient norm using boostr's optimized GPU-accumulating function
            let grad_norm_start = if profile { Some(Instant::now()) } else { None };
            let grad_norm = crate::prelude::compute_grad_norm(&grads)? as f64;
            if let Some(start) = grad_norm_start {
                eprintln!("[PROFILE] grad_norm_compute: {:.3}ms", start.elapsed().as_secs_f64() * 1000.0);
            }

            // DEBUG: Print raw gradient norm BEFORE any clipping
            if std::env::var("DEBUG_RAW_GRAD").is_ok() {
                eprintln!(
                    "[RAW GRAD NORM] step={} raw_grad_norm={:.6}",
                    self.global_step + 1,
                    grad_norm
                );
            }

            // DEBUG: Print top gradient contributors when DEBUG_MLA_MOE or DEBUG_GRAD is set
            // Disabled by default as it causes 121 GPU-CPU syncs (expensive)
            if std::env::var("DEBUG_MLA_MOE").is_ok() || std::env::var("DEBUG_GRAD").is_ok() {
                self.print_gradient_breakdown(&grads);
            }

            // 2. Skip step if gradient norm is invalid (inf/NaN)
            if grad_norm.is_infinite() || grad_norm.is_nan() {
                eprintln!(
                    "[GradClip] Skipping step {} - gradient norm: {}",
                    self.global_step + 1,
                    grad_norm
                );
                // Identify which parameters have problematic gradients
                self.diagnose_bad_gradients(&grads);
                last_grad_norm = grad_norm;
            } else {
                // 3. Global gradient clipping: scale gradients if norm > max_grad_norm
                let max_grad_norm = self.config.max_grad_norm;
                let grads = if grad_norm > max_grad_norm {
                    let scale = (max_grad_norm / (grad_norm + 1e-6)) as f32;
                    let mut scaled = GradStore::new();
                    for (id, grad) in grads.iter() {
                        scaled.insert(*id, grad.scale(scale)?);
                    }
                    scaled
                } else {
                    grads
                };

                // 4. Optional: Clip to target_grad_norm (only scale DOWN if norm exceeds target)
                // This is proper "norm clipping" - preserves gradient direction while preventing explosion
                // Unlike normalization, this does NOT scale up small gradients
                let grads = if let Some(target) = self.config.target_grad_norm {
                    let current_norm = grad_norm.min(max_grad_norm);
                    // Only clip if norm exceeds target (never scale up)
                    if current_norm > target && current_norm > 1e-8 {
                        let scale = (target / current_norm) as f32;
                        let mut clipped = GradStore::new();
                        for (id, grad) in grads.iter() {
                            clipped.insert(*id, grad.scale(scale)?);
                        }
                        clipped
                    } else {
                        grads
                    }
                } else {
                    grads
                };

                // 5. Apply gradients via optimizer (LR already set at start of step)
                let opt_start = if profile { Some(Instant::now()) } else { None };
                self.optimizer.step(&grads, self.varmap);
                if let Some(start) = opt_start {
                    eprintln!("[PROFILE] optimizer.step: {:.3}ms", start.elapsed().as_secs_f64() * 1000.0);
                }

                // Report final grad norm (after clipping)
                // With norm clipping, report the actual norm (capped at target if it was clipped)
                last_grad_norm = if let Some(target) = self.config.target_grad_norm {
                    grad_norm.min(target).min(max_grad_norm)
                } else {
                    grad_norm.min(max_grad_norm)
                };
            }
        }

        // Store gradient norm for monitoring/logging (no longer used for pre-scaling)
        self.prev_grad_norm = Some(last_grad_norm);

        if let Some(start) = step_start {
            eprintln!("[PROFILE] total_step: {:.3}ms\n", start.elapsed().as_secs_f64() * 1000.0);
        }

        self.global_step += 1;
        Ok((total_loss_for_logging / accum_steps as f64, last_grad_norm))
    }

    /// Get the current effective learning rate (accounting for warmup)
    /// Call AFTER global_step has been incremented (i.e., after training_step)
    fn current_learning_rate(&self) -> f64 {
        if self.warmup_steps > 0 && self.global_step <= self.warmup_steps {
            // During warmup: LR = base_lr * step / warmup_steps
            self.base_learning_rate * (self.global_step as f64 / self.warmup_steps as f64)
        } else {
            // After warmup: full base learning rate
            self.base_learning_rate
        }
    }

    pub fn fit<D: DataLoader>(
        &mut self,
        model: &crate::model::LitGPT,
        data_loader: &mut D,
    ) -> Result<()> {
        // Save blazr-compatible config.json if it doesn't exist
        // This enables the checkpoint to be used directly with blazr inference server
        self.save_blazr_config_if_needed();

        if self.headless {
            self.fit_headless(model, data_loader)
        } else {
            self.fit_normal(model, data_loader)
        }
    }

    /// Save blazr-compatible config.json to checkpoint directory (once only)
    fn save_blazr_config_if_needed(&self) {
        // Create checkpoint directory if it doesn't exist
        if !Path::new(&self.config.checkpoint_dir).exists() {
            if let Err(e) = std::fs::create_dir_all(&self.config.checkpoint_dir) {
                if !self.headless {
                    eprintln!("Warning: Failed to create checkpoint directory: {}", e);
                }
                return;
            }
        }

        // Only save if config.json doesn't already exist
        if !BlazrConfig::exists(&self.config.checkpoint_dir) {
            let blazr_config = BlazrConfig::from_model_config(&self.model_config);
            if let Err(e) = blazr_config.save(&self.config.checkpoint_dir) {
                if !self.headless {
                    eprintln!("Warning: Failed to save blazr config.json: {}", e);
                }
            } else if !self.headless {
                println!("📝 Saved config.json for blazr inference compatibility");
            }
        }
    }

    fn fit_headless<D: DataLoader>(
        &mut self,
        model: &crate::model::LitGPT,
        data_loader: &mut D,
    ) -> Result<()> {
        let tokens_per_step = self.batch_size * self.seq_len * self.config.gradient_accumulation;
        let steps_per_epoch = (self.dataset_size + tokens_per_step - 1) / tokens_per_step;
        let training_start = Instant::now();

        // max_steps=0 means unlimited (train for full epochs)
        let effective_steps = if self.config.max_steps == 0 {
            self.total_steps
        } else {
            self.config.max_steps.min(self.total_steps)
        };

        // Log warmup info if enabled
        if self.warmup_steps > 0 {
            println!(
                "{{\"warmup\": {{\"steps\": {}, \"base_lr\": {:.6e}}}}}",
                self.warmup_steps, self.base_learning_rate
            );
        }

        // Resume support: start from start_step
        for step in self.start_step..effective_steps {
            let step_result = self.training_step(model, data_loader);
            let (loss, grad_norm) = match step_result {
                Ok(result) => result,
                Err(e) => {
                    let err_msg = format!("{}", e);
                    eprintln!(
                        "{{\"error\": \"Training error at step {}\", \"message\": \"{}\"}}",
                        step + 1,
                        err_msg.replace('"', "'")
                    );
                    std::process::exit(1);
                }
            };
            let current_epoch = (step + 1) as f64 / steps_per_epoch as f64;

            // Log at intervals
            if step % self.config.log_interval == 0 || step == self.start_step {
                let elapsed = training_start.elapsed().as_secs_f64();
                // Count only steps done in THIS session (not resumed steps)
                let steps_this_session = step - self.start_step + 1;
                let samples_processed =
                    steps_this_session * self.batch_size * self.config.gradient_accumulation;
                let samples_s = samples_processed as f64 / elapsed;

                // Get VRAM usage via nvidia-smi
                let vram = std::process::Command::new("nvidia-smi")
                    .args(&[
                        "--query-compute-apps=used_memory",
                        "--format=csv,noheader,nounits",
                    ])
                    .output()
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .and_then(|s| s.lines().next().map(|l| l.trim().to_string()))
                    .unwrap_or_else(|| "0".to_string());

                println!("{{\"step\": {}, \"loss\": {:.4}, \"grad_norm\": {:.4}, \"learning_rate\": {:.6e}, \"epoch\": {:.2}, \"samples/s\": {:.2}, \"vram\": \"{}\"}}",
                    step + 1, loss, grad_norm, self.current_learning_rate(), current_epoch, samples_s, vram);
            }
        }

        // Always save final checkpoint
        let prefix = self.model_config.checkpoint_prefix();
        println!(
            "Saving final checkpoint to {}/{}_final.safetensors",
            self.config.checkpoint_dir, prefix
        );
        self.save_checkpoint_named("final")?;

        println!("{{\"status\": \"complete\"}}");

        // Force exit to avoid CUDA cleanup hang (boostr/CubeCL resources)
        std::process::exit(0);
    }

    fn fit_normal<D: DataLoader>(
        &mut self,
        model: &crate::model::LitGPT,
        data_loader: &mut D,
    ) -> Result<()> {
        let tokens_per_step = self.batch_size * self.seq_len * self.config.gradient_accumulation;
        let steps_per_epoch = (self.dataset_size + tokens_per_step - 1) / tokens_per_step;

        // Respect max_steps limit (0 = unlimited, train for full epochs)
        let effective_steps = if self.config.max_steps == 0 {
            self.total_steps
        } else {
            self.config.max_steps.min(self.total_steps)
        };

        // MultiProgress: scrollable logs above, sticky progress bar below
        let mp = MultiProgress::new();

        // Progress bar style - ETA calculated manually using rolling average
        let sty =
            ProgressStyle::with_template("{bar:40.cyan/dim} {pos:>7}/{len:7} [{elapsed}<{msg}")
                .unwrap()
                .progress_chars("━━─");

        let pb = mp.add(ProgressBar::new(effective_steps as u64));
        pb.set_style(sty);

        // Resume support - this goes to scrollable area
        if self.start_step > 0 {
            pb.set_position(self.start_step as u64);
            mp.println(format!("📍 Resuming from step {}", self.start_step))
                .ok();
        }

        // Log warmup info if enabled
        if self.warmup_steps > 0 {
            mp.println(format!(
                "🔥 Warmup: {} steps, base LR: {:.6e}",
                self.warmup_steps, self.base_learning_rate
            ))
            .ok();
        }

        let mut last_loss = 0.0;
        let mut last_grad_norm = 0.0;

        // Rolling average for it/s calculation (last N step times)
        const ROLLING_WINDOW: usize = 5;
        let mut step_times: VecDeque<f64> = VecDeque::with_capacity(ROLLING_WINDOW);
        let mut step_start = Instant::now();

        for step in self.start_step..effective_steps {
            let step_result = self.training_step(model, data_loader);

            // Handle errors explicitly so TUI shows them properly
            let (loss, grad_norm) = match step_result {
                Ok(result) => result,
                Err(e) => {
                    // Clear progress bar and show error
                    pb.finish_and_clear();
                    let err_msg = format!("{}", e);
                    mp.println(format!("\n❌ Training error at step {}:", step + 1))
                        .ok();

                    // Check for common CUDA errors
                    if err_msg.contains("out of memory")
                        || err_msg.contains("CUDA")
                        || err_msg.contains("cuda")
                    {
                        mp.println(format!("   🔥 CUDA Error: {}", err_msg)).ok();
                        mp.println(
                            "   💡 Try: reduce batch_size, seq_len, or use gradient accumulation"
                                .to_string(),
                        )
                        .ok();
                    } else if err_msg.contains("NaN") || err_msg.contains("nan") {
                        mp.println(format!("   🔥 Numerical Error: {}", err_msg))
                            .ok();
                        mp.println("   💡 Try: lower learning rate or check data".to_string())
                            .ok();
                    } else {
                        mp.println(format!("   🔥 Error: {}", err_msg)).ok();
                    }

                    // Force exit - CUDA cleanup can hang
                    mp.println("".to_string()).ok();
                    std::process::exit(1);
                }
            };
            last_loss = loss;
            last_grad_norm = grad_norm;

            // Track step time for rolling average
            let step_duration = step_start.elapsed().as_secs_f64();
            step_times.push_back(step_duration);
            if step_times.len() > ROLLING_WINDOW {
                step_times.pop_front();
            }
            step_start = Instant::now();

            let current_epoch = (step + 1) as f64 / steps_per_epoch as f64;

            // Calculate samples/s as rolling average of recent steps
            // Each step processes batch_size * gradient_accumulation samples
            let avg_step_time: f64 = step_times.iter().sum::<f64>() / step_times.len() as f64;
            let samples_per_step = (self.batch_size * self.config.gradient_accumulation) as f64;
            let samples_s = if avg_step_time > 0.0 {
                samples_per_step / avg_step_time
            } else {
                0.0
            };

            // Silent log to file at intervals (no terminal output for steps)
            if step % self.config.log_interval == 0 || step == 0 {
                let step_json = serde_json::json!({
                    "step": step + 1,
                    "loss": loss,
                    "grad_norm": grad_norm,
                    "learning_rate": self.current_learning_rate(),
                    "epoch": current_epoch,
                });
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.steps_json_path)
                {
                    let _ = writeln!(file, "{}", step_json);
                }
            }

            // Calculate ETA based on rolling average
            let remaining_steps = effective_steps - step - 1;
            let eta_secs = avg_step_time * remaining_steps as f64;
            let eta_str = format_duration(eta_secs);

            // Update progress bar (sticky at bottom)
            pb.set_position((step + 1) as u64);
            pb.set_message(format!(
                "{}] loss: {:.4} | grad: {:.4} | {:.2} samples/s",
                eta_str, loss, grad_norm, samples_s
            ));

            // Checkpoint messages go to scrollable area above
            // Only save if we've done at least one new step (not immediately after resume)
            if step > self.start_step && step % self.config.save_interval == 0 {
                mp.println(format!("💾 Checkpoint saved: step {}", step))
                    .ok();
                self.save_checkpoint(step)?;
            }
        }

        pb.finish_with_message(format!(
            "loss: {:.4} | grad: {:.4} | done ✓",
            last_loss, last_grad_norm
        ));

        // Final messages
        let prefix = self.model_config.checkpoint_prefix();
        mp.println(format!(
            "\n💾 Saving final checkpoint to {}/{}_final.safetensors",
            self.config.checkpoint_dir, prefix
        ))
        .ok();
        self.save_checkpoint_named("final")?;

        mp.println(format!(
            "✅ Training complete | final loss: {:.4} | grad: {:.4}",
            last_loss, last_grad_norm
        ))
        .ok();

        Ok(())
    }

    fn save_checkpoint(&self, step: usize) -> Result<()> {
        let prefix = self.model_config.checkpoint_prefix();
        let checkpoint_path = format!(
            "{}/{}_step_{}.safetensors",
            self.config.checkpoint_dir, prefix, step
        );
        if !Path::new(&self.config.checkpoint_dir).exists() {
            std::fs::create_dir_all(&self.config.checkpoint_dir).map_err(Error::wrap)?;
        }
        self.varmap.save(&checkpoint_path)?;

        // Save metadata for auto-resume validation
        if let Some(ref data_path) = self.data_file_path {
            if let Ok(meta) =
                CheckpointMeta::new(step, &self.model_config, data_path, &checkpoint_path)
            {
                if let Err(e) = meta.save(&self.config.checkpoint_dir) {
                    eprintln!("Warning: Failed to save checkpoint metadata: {}", e);
                }
            }
        }

        // Clean up old checkpoints if limit is set
        if self.config.max_checkpoints > 0 {
            self.cleanup_old_checkpoints()?;
        }

        Ok(())
    }

    /// Remove old checkpoints to keep only max_checkpoints most recent
    fn cleanup_old_checkpoints(&self) -> Result<()> {
        let dir = Path::new(&self.config.checkpoint_dir);
        if !dir.exists() {
            return Ok(());
        }

        // Collect all step checkpoints with their step numbers
        // Format: {name}_{seq}_{params}_{dtype}_step_{N}.safetensors
        let mut checkpoints: Vec<(usize, std::path::PathBuf)> = Vec::new();

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                    // Match *_step_N.safetensors (e.g., nano_512_117m_bf16_step_100.safetensors)
                    if filename.contains("_step_") && filename.ends_with(".safetensors") {
                        // Extract step number after "_step_"
                        if let Some(step_part) = filename.split("_step_").last() {
                            if let Some(step_str) = step_part.strip_suffix(".safetensors") {
                                if let Ok(step) = step_str.parse::<usize>() {
                                    checkpoints.push((step, path));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sort by step number (ascending)
        checkpoints.sort_by_key(|(step, _)| *step);

        // Remove oldest checkpoints if we exceed the limit
        let to_remove = checkpoints
            .len()
            .saturating_sub(self.config.max_checkpoints);
        for (step, path) in checkpoints.into_iter().take(to_remove) {
            // Remove checkpoint file
            if let Err(e) = std::fs::remove_file(&path) {
                eprintln!("Warning: Failed to remove old checkpoint {:?}: {}", path, e);
            }
            // Remove associated metadata file
            let meta_path = format!(
                "{}/checkpoint_step_{}.meta.json",
                self.config.checkpoint_dir, step
            );
            let _ = std::fs::remove_file(&meta_path); // Ignore errors for metadata
        }

        Ok(())
    }

    fn save_checkpoint_named(&self, name: &str) -> Result<()> {
        let prefix = self.model_config.checkpoint_prefix();
        let path = format!(
            "{}/{}_{}.safetensors",
            self.config.checkpoint_dir, prefix, name
        );
        if !Path::new(&self.config.checkpoint_dir).exists() {
            std::fs::create_dir_all(&self.config.checkpoint_dir).map_err(Error::wrap)?;
        }
        self.varmap.save(&path)?;
        Ok(())
    }

    /// Compute gradient norm from GradStore
    ///
    /// # Arguments
    /// * `grads` - GradStore from backward pass
    ///
    /// # Returns
    /// Global L2 gradient norm across all parameters
    fn compute_grad_norm_from_grads(&self, grads: &GradStore) -> Result<f64> {
        // Accumulate all squared norms on GPU first, then do a single CPU transfer
        // This avoids 121 GPU-CPU syncs (one per parameter)
        let mut accumulated: Option<Tensor> = None;

        for grad in grads.values() {
            let grad_norm_sq = grad.powf(2.0)?.sum_all()?;
            accumulated = Some(match accumulated {
                None => grad_norm_sq,
                Some(acc) => acc.add(&grad_norm_sq)?,
            });
        }

        // Single GPU->CPU transfer at the end
        let total_norm_sq = match accumulated {
            Some(tensor) => tensor.to_vec0::<f32>()? as f64,
            None => 0.0,
        };

        Ok(total_norm_sq.sqrt())
    }

    /// Print gradient breakdown by parameter (for debugging)
    /// Shows which parameters contribute most to the global gradient norm
    fn print_gradient_breakdown(&self, grads: &GradStore) {
        let mut param_norms: Vec<(String, f64, usize)> = Vec::new();
        let mut total_norm_sq = 0.0f64;

        for (name, tensor) in self.varmap.all_vars_named() {
            if let Some(grad) = grads.get(tensor.id()) {
                if let Ok(norm_sq) = grad
                    .powf(2.0)
                    .and_then(|g| g.sum_all())
                    .and_then(|g| g.to_vec0::<f32>())
                {
                    let norm = (norm_sq as f64).sqrt();
                    let num_params = tensor.elem_count();
                    param_norms.push((name.clone(), norm, num_params));
                    total_norm_sq += norm_sq as f64;
                }
            }
        }

        let total_norm = total_norm_sq.sqrt();

        // Sort by norm (largest first)
        param_norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // If DEBUG_GRAD is set, print ALL gradients sorted by name for easy comparison
        if std::env::var("DEBUG_GRAD").is_ok() {
            eprintln!("\n[GRAD BREAKDOWN - FULL] Total norm: {:.6}", total_norm);
            eprintln!("{}", "=".repeat(80));

            // Sort by name for easy comparison with PyTorch
            let mut by_name = param_norms.clone();
            by_name.sort_by(|a, b| a.0.cmp(&b.0));

            for (name, norm, num_params) in by_name.iter() {
                eprintln!("{:<60}: {:.6e}  (params={})", name, norm, num_params);
            }
            eprintln!("{}", "=".repeat(80));
            eprintln!("TOTAL GRADIENT NORM: {:.6}\n", total_norm);
        } else {
            eprintln!("[GRAD BREAKDOWN] Total norm: {:.4e}", total_norm);
            eprintln!("  Top 10 gradient contributors:");
            for (name, norm, num_params) in param_norms.iter().take(10) {
                let pct = (norm * norm / total_norm_sq * 100.0) as f32;
                eprintln!(
                    "    {}: norm={:.4e}, params={}, contrib={:.1}%",
                    name, norm, num_params, pct
                );
            }
        }
    }

    /// Diagnose which parameters have inf/NaN gradients
    /// Prints the parameter name and gradient norm for problematic parameters
    fn diagnose_bad_gradients(&self, grads: &GradStore) {
        let mut bad_params: Vec<(String, f64)> = Vec::new();

        for (name, tensor) in self.varmap.all_vars_named() {
            if let Some(grad) = grads.get(tensor.id()) {
                if let Ok(norm_sq) = grad
                    .powf(2.0)
                    .and_then(|g| g.sum_all())
                    .and_then(|g| g.to_vec0::<f32>())
                {
                    let norm = (norm_sq as f64).sqrt();
                    if norm.is_infinite() || norm.is_nan() || norm > 1e6 {
                        bad_params.push((name.clone(), norm));
                    }
                }
            }
        }

        // Sort by norm (largest first) and show top offenders
        bad_params.sort_by(|a, b| {
            // NaN/Inf first, then by magnitude
            match (
                a.1.is_nan() || a.1.is_infinite(),
                b.1.is_nan() || b.1.is_infinite(),
            ) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal),
            }
        });

        if !bad_params.is_empty() {
            eprintln!("  Problematic gradients (top 5):");
            for (name, norm) in bad_params.iter().take(5) {
                eprintln!("    {} : grad_norm={:.2e}", name, norm);
            }
        }
    }

    /// Gradient clipping via norm-based scaling
    ///
    /// # Implementation Note
    /// We implement true gradient clipping by:
    /// 1. Computing the global L2 norm from all gradients
    /// 2. If norm > max_norm, scale all gradients by (max_norm / norm)
    /// 3. Return the clipped gradients
    ///
    /// This prevents gradient explosion on the current step, unlike loss pre-scaling
    /// which has a one-step lag.
    ///
    /// # Arguments
    /// * `grads` - GradStore from backward pass
    /// * `max_norm` - Maximum allowed gradient norm (typically 1.0)
    /// Process gradients with full gradient control pipeline:
    /// 1. Per-element clipping (max_grad_value) - clips individual values to [-max, +max]
    /// 2. Global norm clipping (max_grad_norm) - scales entire gradient if ||grad|| > max
    /// 3. Gradient normalization (target_grad_norm) - normalizes to target norm
    ///
    /// # Returns
    /// Tuple of (processed_GradStore, original_grad_norm, should_skip_step)
    ///
    /// Note: Currently unused for single-GPU training (FusedAdamW has built-in clamping),
    /// but kept for distributed training path.
    #[allow(dead_code)]
    fn process_gradients(
        &self,
        grads: GradStore,
        max_grad_norm: f64,
        target_grad_norm: Option<f64>,
        max_grad_value: Option<f64>,
    ) -> Result<(GradStore, f64, bool)> {
        // Step 1: Per-element clipping (if enabled)
        let grads = if let Some(max_val) = max_grad_value {
            let mut clipped = GradStore::new();
            for (id, grad) in grads.iter() {
                let clamped = grad.clamp(-(max_val as f32), max_val as f32)?;
                clipped.insert(*id, clamped);
            }
            clipped
        } else {
            grads
        };

        // Compute gradient norm after per-element clipping
        let total_norm = self.compute_grad_norm_from_grads(&grads)?;

        // If gradient norm is inf/nan, skip this optimizer step
        if total_norm.is_infinite() || total_norm.is_nan() {
            eprintln!(
                "[GradClip] WARN: Gradient norm is {}, skipping step",
                total_norm
            );
            // Identify which parameters have problematic gradients
            for (name, tensor) in self.varmap.all_vars_named() {
                if let Some(grad) = grads.get(tensor.id()) {
                    if let Ok(norm_sq) = grad
                        .powf(2.0)
                        .and_then(|g| g.sum_all())
                        .and_then(|g| g.to_vec0::<f32>())
                    {
                        let norm = (norm_sq as f64).sqrt();
                        if norm.is_infinite() || norm.is_nan() || norm > 1e6 {
                            eprintln!("  └─ {}: grad_norm={:.2e}", name, norm);
                        }
                    }
                }
            }
            return Ok((grads, total_norm, true));
        }

        // Step 2: Global norm clipping - clips if ||grad|| > max_grad_norm
        let (grads, clipped) = if total_norm > max_grad_norm {
            let scale = (max_grad_norm / (total_norm + 1e-6)) as f32;
            let mut scaled = GradStore::new();
            for (id, grad) in grads.iter() {
                scaled.insert(*id, grad.scale(scale)?);
            }
            (scaled, true)
        } else {
            (grads, false)
        };

        // Step 3: Gradient norm clipping (if enabled) - clips to target_grad_norm
        // After max_grad_norm clipping, the effective norm is min(total_norm, max_grad_norm)
        // We only scale DOWN if norm exceeds target (proper norm clipping, not normalization)
        let grads = if let Some(target) = target_grad_norm {
            // Compute current norm after clipping
            let current_norm = if clipped { max_grad_norm } else { total_norm };
            // Only clip if norm exceeds target (never scale up small gradients)
            if current_norm > target && current_norm > 1e-8 {
                let scale = (target / current_norm) as f32;
                let mut clipped_grads = GradStore::new();
                for (id, grad) in grads.iter() {
                    clipped_grads.insert(*id, grad.scale(scale)?);
                }
                clipped_grads
            } else {
                grads
            }
        } else {
            grads
        };

        // Compute the FINAL gradient norm after all processing (clipping + normalization)
        // This is what's actually applied to the optimizer
        let final_norm = self.compute_grad_norm_from_grads(&grads)?;

        Ok((grads, final_norm, false))
    }
}

/// Format seconds into human-readable duration (e.g., "5m32s", "2h14m", "3d05h")
fn format_duration(secs: f64) -> String {
    let total_secs = secs.round() as u64;
    let days = total_secs / 86400;
    let hours = (total_secs % 86400) / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;

    if days > 0 {
        format!("{}d{:02}h", days, hours)
    } else if hours > 0 {
        format!("{}h{:02}m", hours, minutes)
    } else if minutes > 0 {
        format!("{}m{:02}s", minutes, seconds)
    } else {
        format!("{}s", seconds)
    }
}
