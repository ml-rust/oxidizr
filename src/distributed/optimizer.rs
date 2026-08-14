//! Distributed AdamW optimizer adapter
//!
//! The distributed path applies explicitly synchronized gradients, but the
//! optimizer math and state management are owned by boostr's AdamW. Keeping
//! `TensorId` keys intact is mandatory: AdamW stores moments by `TensorId`, so
//! any re-keying would silently orphan optimizer state.

use crate::prelude::{DType, Result, Tensor};
use boostr::autograd::GradStore;
use boostr::ops::{FusedOptimizerOps, TensorOps};
use boostr::optimizer::{AdamW as BoostrAdamW, AdamWConfig as BoostrAdamWConfig};
use boostr::runtime::{Runtime, RuntimeClient};
use boostr::tensor::TensorId;
use boostr::ScalarOps;
use std::collections::HashMap;

/// AdamW optimizer configuration used by oxidizr's distributed adapter.
///
/// This mirrors boostr's `AdamWConfig` and converts into it at construction;
/// all optimizer behavior itself delegates to boostr.
#[derive(Debug, Clone)]
pub struct AdamWConfig {
    /// Learning rate
    pub lr: f64,
    /// Beta1 (first moment decay)
    pub beta1: f64,
    /// Beta2 (second moment decay)
    pub beta2: f64,
    /// Epsilon for numerical stability
    pub eps: f64,
    /// Weight decay coefficient
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        let config = BoostrAdamWConfig::default();
        Self {
            lr: config.lr,
            beta1: config.beta1,
            beta2: config.beta2,
            eps: config.eps,
            weight_decay: config.weight_decay,
        }
    }
}

impl AdamWConfig {
    /// Create config from learning rate with boostr's default betas/eps/decay.
    pub fn with_lr(lr: f64) -> Self {
        Self {
            lr,
            ..Default::default()
        }
    }
}

impl From<AdamWConfig> for BoostrAdamWConfig {
    fn from(config: AdamWConfig) -> Self {
        Self {
            lr: config.lr,
            beta1: config.beta1,
            beta2: config.beta2,
            eps: config.eps,
            weight_decay: config.weight_decay,
        }
    }
}

/// Distributed AdamW optimizer
///
/// This is a thin adapter around `boostr::optimizer::AdamW`. It accepts the
/// optimizer-owned `HashMap<TensorId, Tensor<R>>` used by the migrated trainer
/// and a `GradStore<R>` returned by the synchronizer. The `TensorId` keys are
/// passed through unchanged to keep boostr's moment state stable across steps.
pub struct DistributedAdamW<R>
where
    R: Runtime<DType = DType>,
{
    optimizer: BoostrAdamW<R>,
}

impl<R> DistributedAdamW<R>
where
    R: Runtime<DType = DType>,
{
    /// Create a new DistributedAdamW optimizer.
    ///
    /// `params` is accepted to match the trainer's construction path and to make
    /// the ownership model explicit, but boostr AdamW initializes per-parameter
    /// moments lazily on the first step using these stable keys.
    pub fn new(params: &HashMap<TensorId, Tensor<R>>, config: AdamWConfig) -> Result<Self> {
        let _ = params;
        Ok(Self {
            optimizer: BoostrAdamW::new(config.into()),
        })
    }

    /// Create with just a learning rate (using boostr's default config)
    #[allow(dead_code)]
    pub fn with_lr(params: &HashMap<TensorId, Tensor<R>>, lr: f64) -> Result<Self> {
        Self::new(params, AdamWConfig::with_lr(lr))
    }

    /// Perform an optimizer step with synchronized gradients.
    ///
    /// Both `params` and `synced_grads` must use the same stable `TensorId` keys
    /// produced by the model's trainable-parameter registration.
    pub fn step<C>(
        &mut self,
        client: &C,
        params: &mut HashMap<TensorId, Tensor<R>>,
        synced_grads: &GradStore<R>,
    ) -> Result<()>
    where
        C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + FusedOptimizerOps<R>,
    {
        self.optimizer.step(client, params, synced_grads)
    }

    /// Get the current step count
    #[allow(dead_code)]
    pub fn step_count(&self) -> usize {
        self.optimizer.timestep() as usize
    }

    /// Get the learning rate
    #[allow(dead_code)]
    pub fn learning_rate(&self) -> f64 {
        self.optimizer.config().lr
    }

    /// Set the learning rate (for learning rate scheduling)
    #[allow(dead_code)]
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.optimizer.set_lr(lr);
    }
}

/// Compute or clip gradient norm by delegating to boostr's gradient utility.
pub fn compute_grad_norm<R, C>(client: &C, grads: &mut GradStore<R>, max_norm: f64) -> Result<f64>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R>,
{
    boostr::optimizer::clip_grad_norm(client, grads, max_norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adamw_config_default() {
        let config = AdamWConfig::default();
        let boostr_default = BoostrAdamWConfig::default();
        assert_eq!(config.lr, boostr_default.lr);
        assert_eq!(config.beta1, boostr_default.beta1);
        assert_eq!(config.beta2, boostr_default.beta2);
    }

    #[test]
    fn test_adamw_config_with_lr() {
        let config = AdamWConfig::with_lr(1e-4);
        let boostr_default = BoostrAdamWConfig::default();
        assert_eq!(config.lr, 1e-4);
        assert_eq!(config.beta1, boostr_default.beta1);
    }
}
