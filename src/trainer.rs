use crate::checkpoint_set::{
    apply_optimizer_resume, remove_old_checkpoints, write_checkpoint_set, CheckpointLabel,
    CheckpointSave, NameMaps, OptimizerResume,
};
use crate::config::{OxidizrModelConfig, TrainerConfig};
use crate::data::{DataLoader, DataPosition, RowLayout};
use crate::loss_scale::{resolve_loss_scale, restore_loss_scale, weighted_loss, LossScaler};
use crate::oom_guard::{oom_guard, Pass};
use crate::prelude::{DType, Error, FusedAdamW, Result, Tensor};
use crate::schedule::LrPlan;
use crate::step_loss::StepWeighting;
use crate::weight_decay::{DecayPlan, WeightDecaySpec};
use boostr::autograd::{backward_wrt, var_add, var_mul_scalar, var_reshape, GradStore, Var};
use boostr::ops::{FusedOptimizerOps, TensorOps};
use boostr::optimizer::{AdamWConfig, GradScalerState};
use boostr::readback::scalar_f32;
use boostr::runtime::{Runtime, RuntimeClient};
use boostr::tensor::TensorId;
use boostr::{ReduceOps, TypeConversionOps};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

/// Run-specific metadata needed to construct a `LitTrainer`.
///
/// Bundled into a single struct (rather than passed as separate
/// `LitTrainer::new` arguments) so the constructor's signature stays under
/// clippy's argument-count lint; the fields themselves are unrelated to each
/// other beyond all describing "this particular training run".
pub struct TrainerRunConfig {
    /// Row and batch counts of the loader's shard.
    pub row_layout: RowLayout,
    /// Loader position at `start_step`.
    pub start_position: DataPosition,
    pub batch_size: usize,
    pub steps_json_path: String,
    pub headless: bool,
    pub start_step: usize,
    pub model_config: OxidizrModelConfig,
    pub data_file_path: Option<String>,
    /// Resolved weight decay rule, from `crate::weight_decay::resolve_weight_decay`.
    pub weight_decay: WeightDecaySpec,
}

/// The "LitGPT" style Trainer abstraction.
/// It owns the loop state, the optimizer, and handles the "boring" stuff.
pub struct LitTrainer<'a, R: Runtime<DType = DType>> {
    pub(crate) config: TrainerConfig,
    params: &'a mut HashMap<TensorId, Tensor<R>>,
    parameter_ids: Vec<TensorId>,
    parameter_names: Vec<(String, TensorId)>,
    /// Checkpoint names of the trainable parameters, for optimizer state files.
    name_maps: NameMaps,
    optimizer: FusedAdamW<R>,
    /// The no-decay set the optimizer's parameter group holds.
    decay_plan: DecayPlan,
    client: R::Client,
    /// Training device. Loaded optimizer state goes here.
    device: R::Device,
    global_step: usize,
    // Training metrics
    row_layout: RowLayout,
    pub(crate) batch_size: usize,
    // JSON logging
    pub(crate) steps_json_path: String,
    // UI mode
    pub(crate) headless: bool,
    // Gradient clipping state (for loss pre-scaling heuristic)
    prev_grad_norm: Option<f64>,
    // Resume support
    pub(crate) start_step: usize,
    /// Loader position at `start_step`; checkpoint positions count from here.
    start_position: DataPosition,
    // Checkpoint metadata
    pub(crate) model_config: OxidizrModelConfig,
    data_file_path: Option<String>,
    /// Warmup, post-warmup schedule and step count of this run.
    pub(crate) lr_plan: LrPlan,
    /// Guard against a silently all-ones or nearly-all-zeros loss mask: the
    /// masked-in fraction is reported once per run, not once per step.
    mask_stats_logged: bool,
    /// Loss of the last completed step, for the run record.
    last_loss: Option<f64>,
    /// Dynamic loss scaling. `Some` exactly when the model trains in f16.
    loss_scaler: Option<LossScaler>,
}

impl<'a, R> LitTrainer<'a, R>
where
    R: Runtime<DType = DType>,
    R::Client: RuntimeClient<R>
        + TensorOps<R>
        + FusedOptimizerOps<R>
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
            row_layout,
            start_position,
            batch_size,
            steps_json_path,
            headless,
            start_step,
            model_config,
            data_file_path,
            weight_decay,
        } = run;

        config.distributed.validate()?;
        let name_maps = NameMaps::new(&parameter_names, &parameter_ids)?;

        // The no-decay group is config, not state: build it before any state load.
        let decay_plan =
            DecayPlan::for_params(weight_decay, &parameter_ids, &*params, name_maps.names())?;
        // beta2 = 0.95 stabilizes sparse MoE expert updates. eps = 1e-6 keeps BF16 finite.
        let optimizer = FusedAdamW::with_param_groups(
            AdamWConfig {
                lr: config.learning_rate,
                beta1: config.adamw_beta1,
                beta2: config.adamw_beta2,
                eps: config.adamw_eps,
                weight_decay: config.weight_decay,
            },
            decay_plan.param_groups(),
        )?;

        // Whole batches of `num_epochs` epochs, `gradient_accumulation` per step.
        let total_steps = row_layout.total_steps(config.gradient_accumulation, config.num_epochs);

        let lr_plan = LrPlan::new(&config, total_steps)?;
        // The final dtype, after any `--dtype` override.
        let loss_scaler = resolve_loss_scale(config.loss_scale.as_ref(), model_config.dtype)?
            .map(LossScaler::new)
            .transpose()?;

        Ok(Self {
            config,
            params,
            parameter_ids,
            parameter_names,
            name_maps,
            optimizer,
            decay_plan,
            client,
            device,
            global_step: start_step,
            row_layout,
            batch_size,
            steps_json_path,
            headless,
            prev_grad_norm: None,
            start_step,
            start_position,
            model_config,
            data_file_path,
            lr_plan,
            mask_stats_logged: false,
            last_loss: None,
            loss_scaler,
        })
    }

    /// Performs one training step (including gradient accumulation)
    /// Returns (loss, grad_norm)
    ///
    /// # Gradient Clipping Strategy
    /// Gradients are clipped in place. `crate::grad_clip::clip_gradients` first
    /// clamps each element to `config.max_grad_value` when that is set. It then
    /// clips to `config.max_grad_norm`. The step then clips, optionally, to the
    /// tighter `config.target_grad_norm`. A step whose gradient norm is inf/NaN
    /// is skipped, not clipped.
    ///
    /// The clips measure the norm over `self.parameter_ids`: the tensors the
    /// optimizer steps, and the only keys the merged grad store holds.
    /// `backward_wrt` still retains the loss node's `dL/dL = 1` seed and every
    /// activation on a path to a trainable parameter, but that per-micro-batch
    /// store is filtered to `parameter_ids` on the way into the merged store
    /// and then dropped.
    ///
    /// # Gradient Accumulation Strategy
    /// Each micro-batch runs its own forward AND backward, and its parameter
    /// gradients are summed into one store. By linearity,
    /// `grad(sum_k scaled_loss_k) == sum_k grad(scaled_loss_k)`, so this
    /// computes what chaining the micro-batch losses into one graph and running
    /// a single backward computed — but only one micro-batch's activations are
    /// alive at a time, instead of `gradient_accumulation` of them.
    ///
    /// Float addition is not associative, so the accumulation ORDER matters:
    /// gradients now land in micro-batch order rather than in the merged
    /// graph's reverse-topological order. A parameter reached from several
    /// micro-batches can therefore differ in its last bits, and the reported
    /// `grad_norm` with it. The reported loss cannot move: it is the same
    /// device `add` over the same scalars in the same micro-batch order.
    ///
    /// # Loss Weighting
    /// The step draws all its micro-batches before the first forward pass.
    /// [`StepWeighting`] then picks the cross-entropy form. A masked step
    /// weights every scored token of the step by `1 / total`. An unmasked step
    /// keeps the per-micro-batch mean. Either way the cross-entropy part of
    /// the reported loss is the per-token mean over the step.
    ///
    /// # Loss Scaling
    /// An f16 run multiplies each backward loss by the loss scale on top of
    /// that weighting. After accumulation, every gradient is widened to f32
    /// and divided by the scale, before any clip. A non-finite norm skips the
    /// optimizer step, so its timestep stays put, and the scale backs off.
    /// The step still draws its batches. The reported loss and norm are unscaled.
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
        // Set this step's learning rate once, before forward, backward and the optimizer.
        self.optimizer.set_lr(self.lr_at(self.global_step));

        #[allow(unused_assignments)]
        let mut last_grad_norm = 0.0;
        let accum_steps = self.config.gradient_accumulation;
        let load_balance_alpha = self.config.load_balance_alpha;

        // Profiling support - set PROFILE_TRAIN=1 to enable detailed timing
        let profile = std::env::var("PROFILE_TRAIN").is_ok();
        let step_start = if profile { Some(Instant::now()) } else { None };

        // Accumulated scaled-loss scalar across micro-batches, held as a plain
        // `Tensor` and never as a `Var`. A `Var` carries its `grad_fn`, and a
        // chain of them pins every micro-batch's saved activations at once -
        // exactly the peak this loop exists to avoid.
        let mut accumulated_loss: Option<Tensor<R>> = None;

        // Parameter gradients summed across micro-batches. Backward runs once
        // per micro-batch and its store is folded in here, so only one
        // micro-batch's graph is ever alive.
        let mut accumulated_grads: GradStore<R> = GradStore::new();

        // Draw every micro-batch of the step before the first forward pass.
        // A masked step weights each scored token by `1 / total`, and `total`
        // needs every micro-batch's host-side scored count. The loader hands
        // out the same batches in the same order, so data positions are
        // unchanged.
        let data_start = if profile { Some(Instant::now()) } else { None };
        let mut batches = Vec::with_capacity(accum_steps);
        for _ in 0..accum_steps {
            batches.push(data_loader.next_batch()?);
        }
        let weighting = StepWeighting::for_batches(&batches)?;
        if let Some(start) = data_start {
            eprintln!(
                "[PROFILE] data_load: {:.3}ms",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }

        // Gradient Accumulation Loop - forward and backward, one micro-batch
        // at a time
        for batch in batches {
            let (input, target, loss_mask) = (batch.inputs, batch.targets, batch.loss_mask);
            let scored = batch.scored;

            // Forward pass (returns logits + auxiliary losses + z-losses).
            let fwd_start = if profile { Some(Instant::now()) } else { None };
            let output = oom_guard(Pass::Forward, self.batch_size, || {
                step_model.forward(&self.client, &input)
            })?;
            if let Some(start) = fwd_start {
                eprintln!(
                    "[PROFILE] forward: {:.3}ms",
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }
            let logits = output.logits;
            let aux_losses = output.aux_losses;
            let z_losses = output.z_losses;

            // Check for NaN in logits (indicates model forward pass issue).
            // `all_finite` reduces on device and transfers one scalar, never the
            // logits themselves. Still gated to step 0 / DEBUG_NAN because the
            // reduction is a full pass over the logits.
            let nan_check_start = if profile { Some(Instant::now()) } else { None };
            if (self.global_step == 0 || std::env::var("DEBUG_NAN").is_ok())
                && !all_finite(&self.client, logits.tensor())?
            {
                eprintln!("\n❌ NaN/Inf in model logits! This indicates:");
                eprintln!("   1. Numerical instability in forward pass");
                eprintln!("   2. RoPE or attention computation overflow");
                eprintln!("   3. Mamba SSM state explosion");
                return Err(Error::TrainingError {
                    reason: "NaN in logits".to_string(),
                });
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

            let mask_flat = match &loss_mask {
                Some(mask) => Some(mask.reshape(&[b * s])?),
                None => None,
            };
            if mask_flat.is_some() && !self.mask_stats_logged {
                self.mask_stats_logged = true;
                // `scored` comes from the loader's host copy of the mask, so
                // this report needs no device read.
                let positions = b * s;
                eprintln!(
                    "[loss-mask] masked loss scoring on: {scored}/{positions} positions \
                     masked in ({:.2}%) in the first batch",
                    scored as f64 * 100.0 / positions.max(1) as f64
                );
            }
            // `None`: this micro-batch scores no position and adds no CE term.
            let ce_loss = weighting.ce_term(
                &self.client,
                &logits_flat,
                &target_flat,
                mask_flat.as_ref(),
                scored,
            )?;

            // NaN checking is expensive (GPU->CPU sync) - only do on first step or when DEBUG_NAN is set
            let check_nan = self.global_step == 0 || std::env::var("DEBUG_NAN").is_ok();

            // Check for NaN in CE loss before combining (only when check_nan is true)
            if let (true, Some(ce_loss)) = (check_nan, &ce_loss) {
                let ce_loss_val = scalar_f32(&self.client, ce_loss.tensor())? as f64;
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
                    loss = Some(add_loss_term(loss, scaled_aux, &self.client)?);
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
                    loss = Some(add_loss_term(loss, scaled_z, &self.client)?);
                }
            }

            // A micro-batch with no scored position and no auxiliary loss has
            // nothing to differentiate.
            let Some(loss) = loss else {
                continue;
            };

            // Check combined loss for NaN (only when check_nan is true)
            if check_nan {
                let total_loss_val = scalar_f32(&self.client, loss.tensor())? as f64;
                if total_loss_val.is_nan() || total_loss_val.is_infinite() {
                    eprintln!("\n❌ NaN/Inf in combined loss!");
                    return Err(Error::TrainingError {
                        reason: "NaN loss detected".to_string(),
                    });
                }
            }

            // Scale loss for gradient accumulation. Under f16 the backward
            // loss also carries the loss scale. The logged loss never does.
            let (logged_loss, scaled_loss) =
                weighted_loss(&loss, accum_steps, self.loss_scaler.as_ref(), &self.client)?;

            // Only the scalar tensor is kept, never the `Var`, so nothing here
            // pins this micro-batch's graph.
            accumulated_loss = Some(accumulate_loss_scalar(
                accumulated_loss,
                &logged_loss,
                &self.client,
            )?);

            // Track loss tensor for later logging (defer GPU->CPU transfer)
            // We'll do the transfer after backward when we need to log anyway
            if let Some(start) = loss_start {
                eprintln!(
                    "[PROFILE] loss_compute: {:.3}ms",
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }

            // ===== Backward for THIS micro-batch =====
            // Linearity: grad(sum_k scaled_loss_k) == sum_k grad(scaled_loss_k).
            // Summing per-micro-batch gradients therefore computes the same
            // thing as one backward over the chained loss, while holding one
            // micro-batch's activations instead of `accum_steps` of them.
            let bwd_start = if profile { Some(Instant::now()) } else { None };
            oom_guard(Pass::Backward, self.batch_size, || {
                // Prune to the trainable parameters. Every other node's gradient is
                // unreadable by the optimizer and would only inflate peak memory -
                // a frozen weight under LoRA costs a full-size gradient per step.
                backward_and_merge(
                    scaled_loss,
                    &self.parameter_ids,
                    &mut accumulated_grads,
                    &self.client,
                )
            })?;
            if let Some(start) = bwd_start {
                eprintln!(
                    "[PROFILE] backward: {:.3}ms",
                    start.elapsed().as_secs_f64() * 1000.0
                );
            }

            // `logits`, `loss`, the aux/z-loss Vars and every `grad_fn` they
            // hold are dropped here, at the end of the iteration. This
            // micro-batch's saved activations are released before the next
            // micro-batch's forward pass allocates anything.
        }

        // Transfer loss to CPU for logging (single sync point)
        let loss_sync_start = if profile { Some(Instant::now()) } else { None };
        // A masked step whose micro-batches score nothing reports 0.
        let total_loss_for_logging = match accumulated_loss {
            Some(acc_loss) => scalar_f32(&self.client, &acc_loss)? as f64 * accum_steps as f64,
            None => weighting.loss_without_terms()?,
        };
        if let Some(start) = loss_sync_start {
            eprintln!(
                "[PROFILE] loss_sync: {:.3}ms",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }

        // ===== AFTER ACCUMULATION LOOP =====
        // Every micro-batch has already run its own backward and folded its
        // parameter gradients into `accumulated_grads`.
        let mut grads = accumulated_grads;

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
            // `Grads retained` counts entries in the merged store, which now
            // holds parameter gradients only: each micro-batch's backward store
            // is filtered to `parameter_ids` before being folded in. It
            // therefore equals `Params with grads` and is bounded by
            // `Total params` — do not compare it against logs from before
            // backward moved inside the accumulation loop, where it also
            // counted every activation on a path to a parameter.
            eprintln!(
                "[GRAD CHECK] Total params: {}, Grads retained: {}, Params with grads: {}",
                total_params, grads_count, params_with_grads
            );
            if !params_without_grads.is_empty() {
                // A warning, never an error: under top-k MoE routing an unrouted
                // expert legitimately receives no gradient on a given step.
                eprintln!(
                    "[GRAD CHECK] Params WITHOUT gradients this step ({}) — expected for \
                     unrouted MoE experts, a bug for anything else:",
                    params_without_grads.len()
                );
                for name in &params_without_grads {
                    eprintln!("  - {}", name);
                }
            }
        }

        // 0. Undo the loss scale before any clip, so every clip and the
        // reported norm see true gradients.
        if let Some(scaler) = &self.loss_scaler {
            scaler.unscale(&self.client, &mut grads, &self.parameter_ids)?;
        }

        // 1. Clip each element to `max_grad_value` (when set), then by global norm.
        // The returned norm is the one the norm clip measured.
        let grad_norm_start = if profile { Some(Instant::now()) } else { None };
        let max_grad_norm = self.config.max_grad_norm;
        let grad_norm = crate::grad_clip::clip_gradients(
            &self.client,
            &mut grads,
            &self.parameter_ids,
            self.config.max_grad_value,
            max_grad_norm,
        )?;
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

        // 2. Skip step if gradient norm is invalid (inf/NaN). Under loss
        // scaling this is the overflow signal: the scale backs off below.
        let overflow = !grad_norm.is_finite();
        if overflow {
            if self.loss_scaler.is_none() {
                eprintln!(
                    "[GradClip] Skipping step {} - gradient norm: {}",
                    self.global_step + 1,
                    grad_norm
                );
                self.diagnose_bad_gradients(&grads);
            }
            last_grad_norm = grad_norm;
        } else {
            // 3. Optional tighter target norm clipping delegates to the same boostr utility.
            let effective_after_max = grad_norm.min(max_grad_norm);
            if let Some(target) = self.config.target_grad_norm {
                if effective_after_max > target && effective_after_max > 1e-8 {
                    crate::prelude::compute_grad_norm(
                        &self.client,
                        &mut grads,
                        &self.parameter_ids,
                        target,
                    )?;
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

        if let Some(scaler) = self.loss_scaler.as_mut() {
            let scale = scaler.scale();
            scaler.record_step(overflow);
            if overflow {
                eprintln!(
                    "[LossScale] Skipping step {}: gradient norm {grad_norm} at loss scale \
                     {scale}. The scale backs off to {}.",
                    self.global_step + 1,
                    scaler.scale()
                );
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
        let loss = total_loss_for_logging / accum_steps as f64;
        self.last_loss = Some(loss);
        Ok((loss, last_grad_norm))
    }

    /// The count of completed optimizer steps, resumed steps included.
    pub fn global_step(&self) -> usize {
        self.global_step
    }

    /// The loss of the last completed step. `None` before the first step.
    pub fn last_loss(&self) -> Option<f64> {
        self.last_loss
    }

    /// Learning rate `training_step` applies in 0-indexed step `step`.
    ///
    /// A pure function of `step`, so a resumed run continues the schedule at
    /// its restored `global_step`.
    pub(crate) fn lr_at(&self, step: usize) -> f64 {
        self.lr_plan.lr_at(step)
    }

    fn assert_stable_parameter_ids(&self, model: &crate::model::LitGPT<R>) -> Result<()> {
        let current_ids: Vec<TensorId> = model
            .trainable_parameters()
            .into_iter()
            .map(|(id, _)| id)
            .collect();

        if current_ids != self.parameter_ids {
            // Name the difference: "the ids changed" alone leaves the operator
            // with a 1.2B-parameter model and no way to tell WHICH parameter
            // moved, and the two failure modes need opposite fixes. A set
            // difference means the rebuild changed which parameters are
            // trainable (a freezing bug); an identical set in a different
            // order means only the traversal order moved.
            let before: HashSet<TensorId> = self.parameter_ids.iter().copied().collect();
            let after: HashSet<TensorId> = current_ids.iter().copied().collect();
            let appeared: Vec<&str> = current_ids
                .iter()
                .filter(|id| !before.contains(id))
                .map(|id| self.parameter_name(*id))
                .collect();
            let vanished: Vec<&str> = self
                .parameter_ids
                .iter()
                .filter(|id| !after.contains(id))
                .map(|id| self.parameter_name(*id))
                .collect();
            let detail = if appeared.is_empty() && vanished.is_empty() {
                format!(
                    "the same {} parameters came back in a different order",
                    current_ids.len()
                )
            } else {
                format!(
                    "{} newly trainable ({}), {} no longer trainable ({})",
                    appeared.len(),
                    Self::name_sample(&appeared),
                    vanished.len(),
                    Self::name_sample(&vanished)
                )
            };
            return Err(Error::TrainingError {
                reason: format!(
                    "model rebuild changed trainable TensorIds; AdamW state would be orphaned: \
                     {detail}"
                ),
            });
        }

        Ok(())
    }

    /// A readable sample of a name list: a 1.2B model can move hundreds of
    /// parameters at once, and the first few plus a count is what an operator
    /// reads.
    fn name_sample(names: &[&str]) -> String {
        const LISTED: usize = 6;
        if names.len() <= LISTED {
            return names.join(", ");
        }
        format!(
            "{}, and {} more",
            names[..LISTED].join(", "),
            names.len() - LISTED
        )
    }

    fn parameter_name(&self, id: TensorId) -> &str {
        self.parameter_names
            .iter()
            .find_map(|(name, param_id)| (*param_id == id).then_some(name.as_str()))
            .unwrap_or("<unnamed>")
    }

    /// Loader position after `completed` optimizer steps of this run.
    ///
    /// Every step draws exactly `gradient_accumulation` batches, including a
    /// step skipped for a non-finite gradient norm.
    fn data_position_after(&self, completed: usize) -> DataPosition {
        let batches = (completed - self.start_step) * self.config.gradient_accumulation;
        self.row_layout.advance(self.start_position, batches)
    }

    /// Fractional epochs of data drawn after `completed` optimizer steps.
    pub(crate) fn epoch_progress(&self, completed: usize) -> f64 {
        let pos = self.data_position_after(completed);
        let rows_per_epoch = self.row_layout.batches_per_epoch() * self.row_layout.batch_size();
        pos.epoch as f64 + pos.row as f64 / rows_per_epoch as f64
    }

    /// Saves the checkpoint set labelled with `step`, the count of completed
    /// optimizer steps. A resume from `step_N` starts at step index `N`.
    /// Its loader starts at the position in the meta.
    ///
    /// Frozen parameters are absent from `self.params`, so the model is their
    /// source. Without it a LoRA run writes an adapter-only checkpoint.
    pub fn save_checkpoint(&self, model: &crate::model::LitGPT<R>, step: usize) -> Result<()> {
        self.write_set(model, CheckpointLabel::Step(step), step)?;
        if self.config.max_checkpoints > 0 {
            remove_old_checkpoints(
                Path::new(&self.config.checkpoint_dir),
                self.config.max_checkpoints,
            );
        }
        Ok(())
    }

    /// Saves the `_final` checkpoint set at the current step.
    pub fn save_final_checkpoint(&self, model: &crate::model::LitGPT<R>) -> Result<()> {
        self.write_set(model, CheckpointLabel::Final, self.global_step)
    }

    /// Writes optimizer state, weights, then the meta that marks the set complete.
    fn write_set(
        &self,
        model: &crate::model::LitGPT<R>,
        label: CheckpointLabel,
        step: usize,
    ) -> Result<()> {
        write_checkpoint_set(CheckpointSave {
            dir: Path::new(&self.config.checkpoint_dir),
            prefix: &self.model_config.checkpoint_prefix(),
            label,
            step,
            position: self.data_position_after(step),
            model_config: &self.model_config,
            data_file_path: self.data_file_path.as_deref(),
            model,
            params: self.params,
            optimizer: &self.optimizer,
            name_maps: &self.name_maps,
            weight_decay: self.decay_plan.spec(),
            loss_scale: self.loss_scale_state(),
        })
    }

    /// Applies the optimizer part of a resume plan to this trainer's optimizer.
    pub fn apply_optimizer_resume(&mut self, plan: &OptimizerResume) -> Result<()> {
        apply_optimizer_resume(&mut self.optimizer, &self.name_maps, &self.device, plan)
    }

    /// Continues the loss scale from a resumed checkpoint's `saved` state.
    ///
    /// See [`restore_loss_scale`] for the cases without a scaler or a state.
    pub fn restore_loss_scale(&mut self, saved: Option<GradScalerState>) -> Result<()> {
        restore_loss_scale(self.loss_scaler.as_mut(), saved, self.start_step)
    }

    /// Loss scale and clean-step count. `None` when the run does not scale.
    pub fn loss_scale_state(&self) -> Option<GradScalerState> {
        self.loss_scaler.as_ref().map(LossScaler::state)
    }

    /// The weight decay rule and the no-decay set of this run.
    pub fn decay_plan(&self) -> &DecayPlan {
        &self.decay_plan
    }

    /// The optimizer's step count. A step skipped for a non-finite norm,
    /// a loss scale overflow included, does not count.
    pub fn optimizer_timestep(&self) -> u64 {
        self.optimizer.timestep()
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

    /// Total gradient norm over the trainable parameters.
    ///
    /// `max_norm = f64::MAX` makes this a pure measurement: the clip branch
    /// never fires, so nothing is rescaled. The norm is over
    /// `self.parameter_ids` because `print_gradient_breakdown` divides each
    /// parameter's own norm² by this total to get its contribution share.
    /// Against a whole-graph total those shares would sum to a small fraction
    /// of 100%, since the store also holds the loss seed and every activation.
    fn compute_grad_norm_from_grads(&self, grads: &GradStore<R>) -> Result<f64> {
        let mut cloned = self.clone_grad_store(grads);
        crate::prelude::compute_grad_norm(&self.client, &mut cloned, &self.parameter_ids, f64::MAX)
    }

    fn grad_norm_for_tensor(&self, id: TensorId, grad: &Tensor<R>) -> Result<f64> {
        let mut one = GradStore::new();
        one.insert(id, grad.clone());
        crate::prelude::compute_grad_norm(&self.client, &mut one, &[id], f64::MAX)
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

/// Report whether every element of `tensor` is finite.
///
/// The whole check runs on device and transfers a single scalar, never the
/// tensor: the training rule is that tensor-sized data never crosses to the
/// host for processing, and a full logits tensor is over a gigabyte.
///
/// Summing is what makes one scalar sufficient. A NaN anywhere makes the sum
/// NaN. A single +/-Inf makes it Inf, and two of opposite sign make it NaN
/// when they meet - either way non-finite, so no infinity can hide. Only an
/// all-finite tensor can produce a finite sum.
///
/// The sum keeps the tensor's own dtype (BF16/F16 under mixed precision), so
/// it is cast to F32 before being read as f32: `Tensor::to_vec`/`item` size
/// their copy from the requested Rust type, not from the tensor's dtype, and
/// reading a BF16 scalar as f32 would over-read the device buffer.
fn all_finite<R, C>(client: &C, tensor: &Tensor<R>) -> Result<bool>
where
    R: Runtime<DType = DType>,
    C: ReduceOps<R> + TypeConversionOps<R>,
{
    let contiguous = tensor.contiguous()?;
    let all_dims: Vec<usize> = (0..contiguous.rank()).collect();
    let total = client.sum(&contiguous, &all_dims, false)?;
    Ok(scalar_f32(client, &total)?.is_finite())
}

/// Fold one micro-batch's scaled loss into the running reported-loss scalar.
///
/// This is the same `add`, over the same operands, in the same micro-batch
/// order that the old chain of loss `Var`s performed on its forward pass, so
/// the logged loss is bit-identical to the merged-graph version. Float addition
/// is not associative: reordering or reassociating this sum changes the number
/// the run reports.
///
/// The running total is a plain `Tensor`, never a `Var`. A `Var` would carry
/// its `grad_fn`, and a chain of them would pin every micro-batch's saved
/// activations at once.
fn accumulate_loss_scalar<R, C>(
    accumulated: Option<Tensor<R>>,
    scaled_loss: &Tensor<R>,
    client: &C,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
{
    match accumulated {
        None => Ok(scaled_loss.clone()),
        Some(acc) => Ok(client.add(&acc, scaled_loss)?),
    }
}

/// Run backward for one micro-batch and fold its parameter gradients into
/// `accumulated`.
///
/// `scaled_loss` is taken BY VALUE, so this function owns the handle it is
/// given and releases it before merging. A `&Var` parameter would leave the
/// caller holding the micro-batch's `grad_fn` chain, which is what made every
/// micro-batch's activations coexist in the first place.
///
/// Only `parameter_ids` are carried forward. Every other entry `backward_wrt`
/// retains belongs to an activation whose `TensorId` is minted fresh each
/// micro-batch, so merging those would pile up gradients nothing can ever read
/// and give back the memory this design saves.
fn backward_and_merge<R, C>(
    scaled_loss: Var<R>,
    parameter_ids: &[TensorId],
    accumulated: &mut GradStore<R>,
    client: &C,
) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
{
    let mut micro_grads = backward_wrt(&scaled_loss, parameter_ids, client)?;
    drop(scaled_loss);

    for id in parameter_ids {
        if let Some(grad) = micro_grads.remove(*id) {
            accumulated.try_accumulate(*id, grad, |existing, new| client.add(&existing, &new))?;
        }
    }

    Ok(())
}

/// Adds `term` to a micro-batch loss that can still be empty.
///
/// An empty loss becomes `term`. Otherwise this is the same `var_add`, in the
/// same order, that the loss chain always ran, so unmasked runs keep their bits.
fn add_loss_term<R, C>(loss: Option<Var<R>>, term: Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    match loss {
        None => Ok(term),
        Some(loss) => Ok(var_add(&loss, &term, client)?),
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

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::autograd::{var_mul, GradFn};
    use boostr::{CpuClient, CpuDevice, CpuRuntime};
    use std::sync::{Arc, Weak};

    fn f32_tensor(values: &[f32], device: &CpuDevice) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(values, &[values.len()], device).unwrap()
    }

    /// One micro-batch's scaled loss: `w * x + w * b`.
    ///
    /// `w` is reached by two paths inside a single graph, so intra-graph
    /// accumulation happens too, and by every micro-batch, so the
    /// cross-micro-batch merge is what decides its final value.
    fn scaled_micro_batch_loss(
        w: &Var<CpuRuntime>,
        b: &Var<CpuRuntime>,
        x: f32,
        scale: f64,
        client: &CpuClient,
        device: &CpuDevice,
    ) -> Var<CpuRuntime> {
        let input = Var::new(f32_tensor(&[x], device), false);
        let wx = var_mul(w, &input, client).expect("w * x");
        let wb = var_mul(w, b, client).expect("w * b");
        let loss = var_add(&wx, &wb, client).expect("w * x + w * b");
        var_mul_scalar(&loss, scale, client).expect("scale by 1/accum_steps")
    }

    /// The whole justification for moving backward inside the accumulation
    /// loop: `grad(sum_k scaled_loss_k) == sum_k grad(scaled_loss_k)`. Running
    /// backward per micro-batch and summing the parameter gradients must equal
    /// one backward over the chained loss, or the memory fix has silently
    /// changed what the run trains.
    #[test]
    fn per_micro_batch_backward_matches_one_backward_over_the_summed_loss() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let w = Var::new(f32_tensor(&[0.75], &device), true);
        let b = Var::new(f32_tensor(&[0.5], &device), true);
        let ids = vec![w.id(), b.id()];
        let inputs = [1.0f32, 2.0, 3.0];
        let scale = 1.0 / inputs.len() as f64;

        // The production path: one backward per micro-batch, gradients merged.
        let mut merged = GradStore::<CpuRuntime>::new();
        for x in inputs {
            let scaled = scaled_micro_batch_loss(&w, &b, x, scale, &client, &device);
            backward_and_merge(scaled, &ids, &mut merged, &client).expect("backward and merge");
        }

        // The reference: one graph, one backward - what the loop used to do.
        let mut chained: Option<Var<CpuRuntime>> = None;
        for x in inputs {
            let scaled = scaled_micro_batch_loss(&w, &b, x, scale, &client, &device);
            chained = Some(match chained {
                None => scaled,
                Some(acc) => var_add(&acc, &scaled, &client).expect("chain the losses"),
            });
        }
        let reference =
            backward_wrt(&chained.expect("a chained loss"), &ids, &client).expect("backward");

        for id in &ids {
            let got: Vec<f32> = merged.get(*id).expect("merged gradient").to_vec();
            let want: Vec<f32> = reference.get(*id).expect("reference gradient").to_vec();
            assert_eq!(got.len(), want.len(), "gradient shape for {id:?}");
            for (g, r) in got.iter().zip(want.iter()) {
                assert!(
                    (g - r).abs() < 1e-6,
                    "gradient for {id:?}: merged {got:?} != single-backward {want:?}"
                );
            }
        }
    }

    /// The reported loss is a sum of scalars, and float addition is not
    /// associative. `accumulate_loss_scalar` must reproduce the old `var_add`
    /// chain's forward value bit for bit - same operands, same order, same op.
    ///
    /// The values are chosen so the order is visible in the result: added
    /// smallest-first the two halves of an ULP carry into `1.0 + 2^-23`, added
    /// largest-first each one rounds away to nothing.
    #[test]
    fn accumulated_loss_scalar_is_bit_identical_to_the_var_add_chain() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let half_ulp = 2f32.powi(-24);
        let values = [half_ulp, half_ulp, 1.0f32];

        let mut via_tensor: Option<Tensor<CpuRuntime>> = None;
        let mut via_var: Option<Var<CpuRuntime>> = None;
        for value in values {
            let scaled = Var::new(f32_tensor(&[value], &device), true);
            via_tensor = Some(
                accumulate_loss_scalar(via_tensor, scaled.tensor(), &client)
                    .expect("accumulate the loss scalar"),
            );
            via_var = Some(match via_var {
                None => scaled,
                Some(acc) => var_add(&acc, &scaled, &client).expect("chain the losses"),
            });
        }

        let tensor_sum: Vec<f32> = via_tensor.expect("a loss scalar").to_vec();
        let var_sum: Vec<f32> = via_var.expect("a loss Var").tensor().to_vec();
        assert_eq!(
            tensor_sum[0].to_bits(),
            var_sum[0].to_bits(),
            "tensor accumulation {} != var_add chain {}",
            tensor_sum[0],
            var_sum[0]
        );
        // Pin the order itself, not just the agreement between the two paths:
        // summing largest-first would give exactly 1.0 here.
        assert_eq!(
            tensor_sum[0].to_bits(),
            (1.0f32 + 2f32.powi(-23)).to_bits(),
            "micro-batch order was not preserved: got {}",
            tensor_sum[0]
        );
    }

    /// The memory win itself. A test that only checks numbers passes with the
    /// leak intact, so this one holds a `Weak` to the innermost node of micro-
    /// batch k's graph and asserts it is dead before micro-batch k+1's forward
    /// pass runs.
    ///
    /// The probe is `w * x`, whose `Arc<dyn GradFn>` is held by the `wx` `Var`
    /// itself AND cloned into every downstream node's `grad_fn`. It can only
    /// die once the entire micro-batch graph - and with it every saved
    /// activation - has been released.
    #[test]
    fn each_micro_batch_graph_dies_before_the_next_forward() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let w = Var::new(f32_tensor(&[0.75], &device), true);
        let b = Var::new(f32_tensor(&[0.5], &device), true);
        let ids = vec![w.id(), b.id()];
        let inputs = [1.0f32, 2.0, 3.0];
        let scale = 1.0 / inputs.len() as f64;

        let mut merged = GradStore::<CpuRuntime>::new();
        let mut previous: Option<Weak<dyn GradFn<CpuRuntime>>> = None;

        for (k, x) in inputs.into_iter().enumerate() {
            if let Some(prev) = &previous {
                assert!(
                    prev.upgrade().is_none(),
                    "micro-batch {} still held its graph when micro-batch {k} started its forward",
                    k - 1
                );
            }

            // Mirrors the loop body of `training_step`: every Var below is an
            // iteration local, so the graph dies at the end of the iteration.
            let input = Var::new(f32_tensor(&[x], &device), false);
            let wx = var_mul(&w, &input, &client).expect("w * x");
            let wb = var_mul(&w, &b, &client).expect("w * b");
            let loss = var_add(&wx, &wb, &client).expect("w * x + w * b");
            let scaled = var_mul_scalar(&loss, scale, &client).expect("scale");

            previous = Some(Arc::downgrade(wx.grad_fn().expect("wx has a grad_fn")));

            backward_and_merge(scaled, &ids, &mut merged, &client).expect("backward and merge");
        }

        let last = previous.expect("a probed graph");
        assert!(
            last.upgrade().is_none(),
            "the final micro-batch's graph outlived the accumulation loop"
        );
    }

    /// A parameter only one micro-batch touches - an unrouted MoE expert on
    /// every other micro-batch - must still hold its gradient after the merge.
    /// A merge that inserted instead of accumulating, or that rebuilt the store
    /// per micro-batch, would drop it.
    #[test]
    fn every_parameter_keeps_a_gradient_across_the_accumulation_loop() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let w = Var::new(f32_tensor(&[0.75], &device), true);
        let b = Var::new(f32_tensor(&[0.5], &device), true);
        let rare = Var::new(f32_tensor(&[2.0], &device), true);
        let ids = vec![w.id(), b.id(), rare.id()];
        let inputs = [1.0f32, 2.0, 3.0];

        let mut merged = GradStore::<CpuRuntime>::new();
        for (k, x) in inputs.into_iter().enumerate() {
            let input = Var::new(f32_tensor(&[x], &device), false);
            let wx = var_mul(&w, &input, &client).expect("w * x");
            let tail = if k == 0 {
                var_mul(&rare, &b, &client).expect("rare * b")
            } else {
                b.clone()
            };
            let loss = var_add(&wx, &tail, &client).expect("w * x + tail");
            let scaled = var_mul_scalar(&loss, 1.0 / inputs.len() as f64, &client).expect("scale");
            backward_and_merge(scaled, &ids, &mut merged, &client).expect("backward and merge");
        }

        for id in &ids {
            assert!(
                merged.get(*id).is_some(),
                "parameter {id:?} received no gradient across the accumulation loop"
            );
        }
        // The merged store holds parameter entries only - this is what
        // `[GRAD CHECK] Grads retained` now counts.
        assert_eq!(merged.len(), ids.len(), "non-parameter gradients were kept");
    }

    #[test]
    fn all_finite_accepts_a_finite_tensor() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let t = f32_tensor(&[1.0, -2.0, 3.5, 0.0], &device);
        assert!(all_finite(&client, &t).expect("check runs"));
    }

    #[test]
    fn all_finite_rejects_a_nan() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let t = f32_tensor(&[1.0, f32::NAN, 3.5], &device);
        assert!(!all_finite(&client, &t).expect("check runs"));
    }

    /// Both signs of infinity, and both together: the +/- pair meets inside the
    /// reduction and becomes NaN, which is still non-finite, so nothing hides.
    #[test]
    fn all_finite_rejects_every_infinity() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        for values in [
            vec![1.0, f32::INFINITY, 3.5],
            vec![1.0, f32::NEG_INFINITY, 3.5],
            vec![f32::INFINITY, f32::NEG_INFINITY],
        ] {
            let t = f32_tensor(&values, &device);
            assert!(
                !all_finite(&client, &t).expect("check runs"),
                "{values:?} must be reported non-finite"
            );
        }
    }

    /// Every scalar the trainer logs (CE loss, combined loss, accumulated loss)
    /// carries the model's compute dtype. Read at a fixed f32 width, a BF16
    /// scalar comes back as reinterpreted bytes - a plausible-looking but wrong
    /// number, not an error - so the loss would be silently garbage.
    #[test]
    fn scalar_f32_reads_a_bf16_scalar_at_its_own_dtype() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let scalar = client
            .cast(&f32_tensor(&[11.5], &device), DType::BF16)
            .expect("cast to bf16");
        let value = scalar_f32(&client, &scalar).expect("readback runs");
        assert!(
            (value - 11.5).abs() < 0.1,
            "expected ~11.5 back from BF16, got {value}"
        );
    }

    /// The BF16 regression: reading the reduction at a fixed f32 width against a
    /// BF16 tensor asks the buffer for twice the bytes it holds. On CUDA that is
    /// the `CUDA_ERROR_INVALID_VALUE` that killed step 0 of every half-precision
    /// run; here it is an over-read of the host buffer. `all_finite` casts the
    /// scalar to F32 first, so both cases below answer correctly.
    #[test]
    fn all_finite_reads_a_bf16_tensor_at_its_own_dtype() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let finite = client
            .cast(&f32_tensor(&[1.0, -2.0, 3.5], &device), DType::BF16)
            .expect("cast to bf16");
        assert_eq!(finite.dtype(), DType::BF16);
        assert!(all_finite(&client, &finite).expect("check runs"));

        let with_nan = client
            .cast(&f32_tensor(&[1.0, f32::NAN, 3.5], &device), DType::BF16)
            .expect("cast to bf16");
        assert!(!all_finite(&client, &with_nan).expect("check runs"));
    }
}
