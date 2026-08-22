use crate::config::{BlazrConfig, CheckpointMeta, OxidizrModelConfig, TrainerConfig};
use crate::data::DataLoader;
use crate::distributed::{
    create_synchronizer, AdamWConfig as DistAdamWConfig, DistributedAdamW, GradientSynchronizer,
};
use crate::prelude::{cross_entropy, DType, Error, FusedAdamW, Result, Tensor};
use boostr::autograd::{backward, var_add, var_mul_scalar, var_reshape, GradStore, Var};
use boostr::ops::{FusedOptimizerOps, TensorOps};
use boostr::optimizer::AdamWConfig;
use boostr::runtime::{Runtime, RuntimeClient};
use boostr::tensor::TensorId;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

/// Run-specific metadata needed to construct a `LitTrainer`.
///
/// Bundled into a single struct (rather than passed as separate
/// `LitTrainer::new` arguments) so the constructor's signature stays under
/// clippy's argument-count lint; the fields themselves are unrelated to each
/// other beyond all describing "this particular training run".
pub struct TrainerRunConfig {
    pub dataset_size: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub steps_json_path: String,
    pub headless: bool,
    pub start_step: usize,
    pub model_config: OxidizrModelConfig,
    pub data_file_path: Option<String>,
}

/// The "LitGPT" style Trainer abstraction.
/// It owns the loop state, the optimizer, and handles the "boring" stuff.
pub struct LitTrainer<'a, R: Runtime<DType = DType>> {
    config: TrainerConfig,
    params: &'a mut HashMap<TensorId, Tensor<R>>,
    parameter_ids: Vec<TensorId>,
    parameter_names: Vec<(String, TensorId)>,
    optimizer: FusedAdamW<R>,
    client: R::Client,
    #[allow(dead_code)] // Kept for potential future use (e.g., device-specific optimizations)
    device: R::Device,
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
    model_config: OxidizrModelConfig,
    data_file_path: Option<String>,
    // Distributed training support
    synchronizer: Option<Box<dyn GradientSynchronizer<R>>>,
    distributed_optimizer: Option<DistributedAdamW<R>>,
    // Learning rate warmup (critical for Mamba SSM parameter alignment)
    warmup_steps: usize,     // Effective warmup steps (0 = disabled)
    base_learning_rate: f64, // Base LR to warmup towards
}

impl<'a, R> LitTrainer<'a, R>
where
    R: Runtime<DType = DType> + crate::distributed::RuntimeDeviceFromOrdinal,
    R::Client: RuntimeClient<R>
        + TensorOps<R>
        + FusedOptimizerOps<R>
        + boostr::ScalarOps<R>
        + boostr::CompareOps<R>
        + boostr::ConvOps<R>
        + boostr::FlashAttentionOps<R>
        + boostr::IndexingOps<R>
        + boostr::NormalizationOps<R>
        + boostr::QuantMatmulOps<R>
        + boostr::RoPEOps<R>
        + boostr::TypeConversionOps<R>,
{
    pub fn new(
        config: TrainerConfig,
        params: &'a mut HashMap<TensorId, Tensor<R>>,
        parameter_ids: Vec<TensorId>,
        parameter_names: Vec<(String, TensorId)>,
        client: R::Client,
        device: R::Device,
        run: TrainerRunConfig,
    ) -> Result<Self> {
        let TrainerRunConfig {
            dataset_size,
            batch_size,
            seq_len,
            steps_json_path,
            headless,
            start_step,
            model_config,
            data_file_path,
        } = run;

        // Init FusedAdamW with configurable SOTA parameters (from optimization.md)
        // - beta2 = 0.95: Lower β₂ stabilizes MoE sparse expert updates
        // - eps = 1e-6: Higher ε for BF16 numerical safety
        // - weight_decay = 0.1: Industry standard (decoupled from LR)
        let optimizer = FusedAdamW::new(AdamWConfig {
            lr: config.learning_rate,
            beta1: config.adamw_beta1,
            beta2: config.adamw_beta2,
            eps: config.adamw_eps,
            weight_decay: config.weight_decay,
        });

        // Calculate total steps from dataset
        let effective_batch_size = batch_size * seq_len * config.gradient_accumulation;
        let steps_per_epoch = dataset_size.div_ceil(effective_batch_size); // Ceiling division
        let total_steps = steps_per_epoch * config.num_epochs;

        // Initialize distributed training components if enabled
        let (synchronizer, distributed_optimizer) = if config.distributed.enabled {
            let sync = create_synchronizer(&config.distributed)?;
            let dist_opt =
                DistributedAdamW::new(params, DistAdamWConfig::with_lr(config.learning_rate))?;
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
            params,
            parameter_ids,
            parameter_names,
            optimizer,
            client,
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
    /// Gradients are clipped in place by `boostr::optimizer::clip_grad_norm`, first
    /// to `config.max_grad_norm` and then, optionally, to the tighter
    /// `config.target_grad_norm`. Steps whose gradient norm is inf/NaN are skipped
    /// entirely rather than clipped. The earlier loss-pre-scaling heuristic (a
    /// workaround for Candle's lack of in-place gradient modification) is gone.
    pub fn training_step<D: DataLoader<R>>(
        &mut self,
        model: &crate::model::LitGPT<R>,
        data_loader: &mut D,
    ) -> Result<(f64, f64)> {
        // Rebuild this step's trainable Vars from the optimizer-owned tensor map.
        // The model migration must implement this using Var::with_id(..., id, true)
        // for every entry, preserving the TensorId keys that AdamW uses for state.
        let step_model = model.rebuild_with_parameters(self.params)?;
        self.assert_stable_parameter_ids(&step_model)?;
        // Apply learning rate warmup at the START of each step (critical for Mamba SSM alignment)
        // This ensures consistent LR throughout forward/backward/optimizer
        if self.warmup_steps > 0 && self.global_step < self.warmup_steps {
            let warmup_ratio = (self.global_step + 1) as f64 / self.warmup_steps as f64;
            let warmup_lr = (self.base_learning_rate * warmup_ratio).max(1e-7);
            self.optimizer.set_lr(warmup_lr);
        } else if self.warmup_steps > 0 && self.global_step == self.warmup_steps {
            // Restore full learning rate after warmup completes
            self.optimizer.set_lr(self.base_learning_rate);
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

        // Accumulated loss Var across micro-batches.
        // We keep the autograd graph intact, then call backward once on the total loss.
        let mut accumulated_loss: Option<Var<R>> = None;

        // Gradient Accumulation Loop - only forward passes, accumulate losses
        for _micro_batch_idx in 0..accum_steps {
            let data_start = if profile { Some(Instant::now()) } else { None };
            let (input, target) = data_loader.next_batch()?;
            if let Some(start) = data_start {
                eprintln!(
                    "[PROFILE] data_load: {:.3}ms",
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }

            // Forward pass (returns logits + auxiliary losses + z-losses)
            // Wrap in catch_unwind to provide user-friendly OOM error
            let fwd_start = if profile { Some(Instant::now()) } else { None };
            let output = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                step_model.forward(&self.client, &input)
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
                            if let Some(size) = size_str.split_whitespace().next() {
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

                        return Err(Error::TrainingError {
                            reason: format!(
                                "GPU out of memory during forward pass{}\n\n\
                                The model requires more VRAM than available. Try:\n\
                                • Reduce batch_size (current: {})\n\
                                • Reduce max_seq_len in config\n\
                                • Increase gradient_accumulation to maintain effective batch size\n\
                                • Use a smaller model configuration\n\
                                • Free GPU memory from other processes (check with: nvidia-smi)",
                                size_info, self.batch_size
                            ),
                        });
                    }

                    std::panic::resume_unwind(panic_info);
                }
            };
            if let Some(start) = fwd_start {
                eprintln!(
                    "[PROFILE] forward: {:.3}ms",
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }
            let logits = output.logits;
            let aux_losses = output.aux_losses;
            let z_losses = output.z_losses;

            // Check for NaN in logits (indicates model forward pass issue)
            // EXPENSIVE: This transfers entire logits tensor to CPU - only do on first step or if DEBUG
            let nan_check_start = if profile { Some(Instant::now()) } else { None };
            if self.global_step == 0 || std::env::var("DEBUG_NAN").is_ok() {
                let logits_tensor = logits.tensor().contiguous()?;
                let logits_vec: Vec<f32> = logits_tensor.to_vec();
                if logits_vec.iter().any(|x| x.is_nan() || x.is_infinite()) {
                    eprintln!("\n❌ NaN/Inf in model logits! This indicates:");
                    eprintln!("   1. Numerical instability in forward pass");
                    eprintln!("   2. RoPE or attention computation overflow");
                    eprintln!("   3. Mamba SSM state explosion");
                    return Err(Error::TrainingError {
                        reason: "NaN in logits".to_string(),
                    });
                }
            }
            if let Some(start) = nan_check_start {
                eprintln!(
                    "[PROFILE] nan_check: {:.3}ms",
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }

            // Calculate Loss (Cross Entropy)
            // Flatten batch/seq dims for cross_entropy: [Batch*Seq, Vocab]
            let loss_start = if profile { Some(Instant::now()) } else { None };
            let (b, s, v) = logits.tensor().dims3()?;
            let logits_flat = var_reshape(&logits, &[b * s, v])?;
            let target_flat = target.reshape(&[b * s])?;

            let ce_loss = cross_entropy(&self.client, &logits_flat, &target_flat)?;

            // NaN checking is expensive (GPU->CPU sync) - only do on first step or when DEBUG_NAN is set
            let check_nan = self.global_step == 0 || std::env::var("DEBUG_NAN").is_ok();

            // Check for NaN in CE loss before combining (only when check_nan is true)
            if check_nan {
                let ce_loss_val = ce_loss.tensor().item::<f32>()? as f64;
                if ce_loss_val.is_nan() || ce_loss_val.is_infinite() {
                    eprintln!("\n❌ NaN/Inf detected in CE loss! This usually indicates:");
                    eprintln!("   1. Learning rate too high (try 1e-4 or lower)");
                    eprintln!("   2. Gradient explosion (sequence too long)");
                    eprintln!("   3. Numerical instability in model");
                    eprintln!("   Current CE loss: {}", ce_loss_val);
                    return Err(Error::TrainingError {
                        reason: "NaN loss detected".to_string(),
                    });
                }
            }

            // Add auxiliary losses (e.g., MoE load balancing) as Vars so the graph is preserved.
            let mut loss = ce_loss;
            if !aux_losses.is_empty() {
                let num_aux = aux_losses.len() as f64;
                for aux_loss in &aux_losses {
                    let loss_t = scalar_loss_var(aux_loss)?;
                    let scaled_aux =
                        var_mul_scalar(&loss_t, load_balance_alpha / num_aux, &self.client)?;
                    loss = var_add(&loss, &scaled_aux, &self.client)?;
                }
            }

            // Add router z-losses (prevents MoE logit explosion in BF16)
            // L_total = L_MLE + α*L_aux + γ*L_z where γ = z_loss_alpha
            let z_loss_alpha = self.config.z_loss_alpha;
            if !z_losses.is_empty() && z_loss_alpha > 0.0 {
                let num_z = z_losses.len() as f64;
                for z_loss in &z_losses {
                    let loss_t = scalar_loss_var(z_loss)?;
                    let scaled_z = var_mul_scalar(&loss_t, z_loss_alpha / num_z, &self.client)?;
                    loss = var_add(&loss, &scaled_z, &self.client)?;
                }
            }

            // Check combined loss for NaN (only when check_nan is true)
            if check_nan {
                let total_loss_val = loss.tensor().item::<f32>()? as f64;
                if total_loss_val.is_nan() || total_loss_val.is_infinite() {
                    eprintln!("\n❌ NaN/Inf in combined loss!");
                    return Err(Error::TrainingError {
                        reason: "NaN loss detected".to_string(),
                    });
                }
            }

            // Scale loss for gradient accumulation
            let scaled_loss = var_mul_scalar(&loss, 1.0 / accum_steps as f64, &self.client)?;

            // Accumulate scaled loss (keeps computation graph for single backward)
            accumulated_loss = Some(match accumulated_loss {
                None => scaled_loss,
                Some(acc) => var_add(&acc, &scaled_loss, &self.client)?,
            });

            // Track loss tensor for later logging (defer GPU->CPU transfer)
            // We'll do the transfer after backward when we need to log anyway
            if let Some(start) = loss_start {
                eprintln!(
                    "[PROFILE] loss_compute: {:.3}ms",
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }
        }

        // Transfer loss to CPU for logging (single sync point)
        let loss_sync_start = if profile { Some(Instant::now()) } else { None };
        if let Some(ref acc_loss) = accumulated_loss {
            // Transfer the accumulated loss value
            total_loss_for_logging = acc_loss.tensor().item::<f32>()? as f64 * accum_steps as f64;
        }
        if let Some(start) = loss_sync_start {
            eprintln!(
                "[PROFILE] loss_sync: {:.3}ms",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }

        // ===== AFTER ACCUMULATION LOOP: Single backward pass =====
        // All micro-batches have been processed, now do backward on accumulated loss

        let total_loss = accumulated_loss.ok_or_else(|| Error::TrainingError {
            reason: "No loss accumulated".to_string(),
        })?;

        // Single backward pass over all micro-batches
        // Wrap in catch_unwind to provide user-friendly OOM error instead of panic
        let bwd_start = if profile { Some(Instant::now()) } else { None };
        let grads = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backward(&total_loss, &self.client)
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
                        if let Some(size) = size_str.split_whitespace().next() {
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

                    return Err(Error::TrainingError {
                        reason: format!(
                            "GPU out of memory during backward pass{}\n\n\
                            The model requires more VRAM than available. Try:\n\
                            • Reduce batch_size (current: {})\n\
                            • Reduce max_seq_len in config\n\
                            • Increase gradient_accumulation to maintain effective batch size\n\
                            • Use a smaller model configuration\n\
                            • Free GPU memory from other processes (check with: nvidia-smi)",
                            size_info, self.batch_size
                        ),
                    });
                }

                // Re-panic for other errors
                std::panic::resume_unwind(panic_info);
            }
        };
        if let Some(start) = bwd_start {
            eprintln!(
                "[PROFILE] backward: {:.3}ms",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }

        // Check parameter/gradient matching at step 1.
        // Gradients must be keyed by the same stable TensorIds used in self.params.
        if self.global_step == 0 {
            let total_params = self.parameter_ids.len();
            let grads_count = grads.len();
            let mut params_with_grads = 0;
            let mut params_without_grads = Vec::new();
            for id in &self.parameter_ids {
                if grads.get(*id).is_some() {
                    params_with_grads += 1;
                } else {
                    params_without_grads.push(self.parameter_name(*id).to_string());
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
            // DISTRIBUTED PATH: Synchronize gradients across GPUs.
            // The distributed module migration must keep gradients keyed by the same TensorIds.
            let mut synced_grads = sync.synchronize_gradients(&grads, self.params)?;

            let grad_norm = crate::prelude::compute_grad_norm(
                &self.client,
                &mut synced_grads,
                self.config.max_grad_norm,
            )?;
            last_grad_norm = grad_norm;

            dist_opt.step(&self.client, self.params, &synced_grads)?;
        } else {
            // SINGLE-GPU PATH
            let mut grads = grads;

            // 1. Clip by global norm using boostr AdamW's gradient utility.
            // The function returns the original norm as its value.
            let grad_norm_start = if profile { Some(Instant::now()) } else { None };
            let max_grad_norm = self.config.max_grad_norm;
            let grad_norm =
                crate::prelude::compute_grad_norm(&self.client, &mut grads, max_grad_norm)?;
            if let Some(start) = grad_norm_start {
                eprintln!(
                    "[PROFILE] grad_norm_compute: {:.3}ms",
                    start.elapsed().as_secs_f64() * 1000.0
                );
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
                self.diagnose_bad_gradients(&grads);
                last_grad_norm = grad_norm;
            } else {
                // 3. Optional tighter target norm clipping delegates to the same boostr utility.
                let effective_after_max = grad_norm.min(max_grad_norm);
                if let Some(target) = self.config.target_grad_norm {
                    if effective_after_max > target && effective_after_max > 1e-8 {
                        crate::prelude::compute_grad_norm(&self.client, &mut grads, target)?;
                    }
                }

                // 4. Apply gradients via optimizer (LR already set at start of step)
                let opt_start = if profile { Some(Instant::now()) } else { None };
                self.optimizer.step(&self.client, self.params, &grads)?;
                if let Some(start) = opt_start {
                    eprintln!(
                        "[PROFILE] optimizer.step: {:.3}ms",
                        start.elapsed().as_secs_f64() * 1000.0
                    );
                }

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
            eprintln!(
                "[PROFILE] total_step: {:.3}ms\n",
                start.elapsed().as_secs_f64() * 1000.0
            );
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

    pub fn fit<D: DataLoader<R>>(
        &mut self,
        model: &crate::model::LitGPT<R>,
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

    fn fit_headless<D: DataLoader<R>>(
        &mut self,
        model: &crate::model::LitGPT<R>,
        data_loader: &mut D,
    ) -> Result<()> {
        let tokens_per_step = self.batch_size * self.seq_len * self.config.gradient_accumulation;
        let steps_per_epoch = self.dataset_size.div_ceil(tokens_per_step);
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
                    .args([
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

    fn fit_normal<D: DataLoader<R>>(
        &mut self,
        model: &crate::model::LitGPT<R>,
        data_loader: &mut D,
    ) -> Result<()> {
        let tokens_per_step = self.batch_size * self.seq_len * self.config.gradient_accumulation;
        let steps_per_epoch = self.dataset_size.div_ceil(tokens_per_step);

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
                            "   💡 Try: reduce batch_size, seq_len, or use gradient accumulation",
                        )
                        .ok();
                    } else if err_msg.contains("NaN") || err_msg.contains("nan") {
                        mp.println(format!("   🔥 Numerical Error: {}", err_msg))
                            .ok();
                        mp.println("   💡 Try: lower learning rate or check data")
                            .ok();
                    } else {
                        mp.println(format!("   🔥 Error: {}", err_msg)).ok();
                    }

                    // Force exit - CUDA cleanup can hang
                    mp.println("").ok();
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

    fn assert_stable_parameter_ids(&self, model: &crate::model::LitGPT<R>) -> Result<()> {
        let current_ids: Vec<TensorId> = model
            .trainable_parameters()
            .into_iter()
            .map(|(id, _)| id)
            .collect();

        if current_ids != self.parameter_ids {
            return Err(Error::TrainingError {
                reason: "model rebuild changed trainable TensorIds; AdamW state would be orphaned"
                    .to_string(),
            });
        }

        Ok(())
    }

    fn parameter_name(&self, id: TensorId) -> &str {
        self.parameter_names
            .iter()
            .find_map(|(name, param_id)| (*param_id == id).then_some(name.as_str()))
            .unwrap_or("<unnamed>")
    }

    fn save_checkpoint(&self, step: usize) -> Result<()> {
        let prefix = self.model_config.checkpoint_prefix();
        let checkpoint_path = format!(
            "{}/{}_step_{}.safetensors",
            self.config.checkpoint_dir, prefix, step
        );
        if !Path::new(&self.config.checkpoint_dir).exists() {
            std::fs::create_dir_all(&self.config.checkpoint_dir)?;
        }

        crate::model::save_checkpoint_from_params(
            &checkpoint_path,
            self.params,
            &self.parameter_names,
        )?;

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
                    if filename.contains("_step_") && filename.ends_with(".safetensors") {
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

        checkpoints.sort_by_key(|(step, _)| *step);

        let to_remove = checkpoints
            .len()
            .saturating_sub(self.config.max_checkpoints);
        for (step, path) in checkpoints.into_iter().take(to_remove) {
            if let Err(e) = std::fs::remove_file(&path) {
                eprintln!("Warning: Failed to remove old checkpoint {:?}: {}", path, e);
            }
            let meta_path = format!(
                "{}/checkpoint_step_{}.meta.json",
                self.config.checkpoint_dir, step
            );
            let _ = std::fs::remove_file(&meta_path);
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
            std::fs::create_dir_all(&self.config.checkpoint_dir)?;
        }
        crate::model::save_checkpoint_from_params(&path, self.params, &self.parameter_names)?;
        Ok(())
    }

    fn clone_grad_store(&self, grads: &GradStore<R>) -> GradStore<R> {
        let mut cloned = GradStore::new();
        for id in grads.keys().copied() {
            if let Some(grad) = grads.get(id) {
                cloned.insert(id, grad.clone());
            }
        }
        cloned
    }

    /// Compute gradient norm by delegating to boostr::optimizer::clip_grad_norm.
    fn compute_grad_norm_from_grads(&self, grads: &GradStore<R>) -> Result<f64> {
        let mut cloned = self.clone_grad_store(grads);
        crate::prelude::compute_grad_norm(&self.client, &mut cloned, f64::MAX)
    }

    fn grad_norm_for_tensor(&self, id: TensorId, grad: &Tensor<R>) -> Result<f64> {
        let mut one = GradStore::new();
        one.insert(id, grad.clone());
        crate::prelude::compute_grad_norm(&self.client, &mut one, f64::MAX)
    }

    /// Print gradient breakdown by parameter (for debugging).
    fn print_gradient_breakdown(&self, grads: &GradStore<R>) {
        let mut param_norms: Vec<(String, f64, usize)> = Vec::new();
        let total_norm = self.compute_grad_norm_from_grads(grads).unwrap_or(0.0);
        let total_norm_sq = total_norm * total_norm;

        for id in &self.parameter_ids {
            if let Some(grad) = grads.get(*id) {
                if let Ok(norm) = self.grad_norm_for_tensor(*id, grad) {
                    let num_params = self
                        .params
                        .get(id)
                        .map(|tensor| tensor.elem_count())
                        .unwrap_or(0);
                    param_norms.push((self.parameter_name(*id).to_string(), norm, num_params));
                }
            }
        }

        param_norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if std::env::var("DEBUG_GRAD").is_ok() {
            eprintln!("\n[GRAD BREAKDOWN - FULL] Total norm: {:.6}", total_norm);
            eprintln!("{}", "=".repeat(80));

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
                let pct = if total_norm_sq > 0.0 {
                    (norm * norm / total_norm_sq * 100.0) as f32
                } else {
                    0.0
                };
                eprintln!(
                    "    {}: norm={:.4e}, params={}, contrib={:.1}%",
                    name, norm, num_params, pct
                );
            }
        }
    }

    /// Diagnose which parameters have inf/NaN gradients.
    fn diagnose_bad_gradients(&self, grads: &GradStore<R>) {
        let mut bad_params: Vec<(String, f64)> = Vec::new();

        for id in &self.parameter_ids {
            if let Some(grad) = grads.get(*id) {
                if let Ok(norm) = self.grad_norm_for_tensor(*id, grad) {
                    if norm.is_infinite() || norm.is_nan() || norm > 1e6 {
                        bad_params.push((self.parameter_name(*id).to_string(), norm));
                    }
                }
            }
        }

        bad_params.sort_by(|a, b| {
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
}

fn scalar_loss_var<R>(loss: &Var<R>) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
{
    if loss.tensor().elem_count() != 1 {
        return Err(Error::TrainingError {
            reason: format!(
                "expected scalar auxiliary loss, got shape {:?}",
                loss.tensor().shape()
            ),
        });
    }

    if loss.tensor().rank() == 0 {
        Ok(loss.clone())
    } else {
        var_reshape(loss, &[]).map_err(Error::from)
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
