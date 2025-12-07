//! Distributed AdamW Optimizer
//!
//! A custom AdamW implementation that works with explicitly provided gradients,
//! enabling its use with synchronized gradients from distributed training.
//!
//! # Why a Custom Optimizer?
//!
//! Candle's built-in AdamW takes a `GradStore` which is opaque and cannot be
//! modified. For distributed training, we need to:
//! 1. Extract gradients from GradStore
//! 2. Synchronize (all-reduce) across GPUs
//! 3. Apply the synchronized gradients
//!
//! This optimizer takes explicit gradient tensors instead of a GradStore,
//! allowing it to work with the synchronized gradients.

use crate::prelude::{DType, Error, Result, Tensor, VarMap};
use std::collections::HashMap;

use super::sync::SyncedGradient;

/// AdamW optimizer configuration
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
        Self {
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

impl AdamWConfig {
    /// Create config from learning rate with default betas
    pub fn with_lr(lr: f64) -> Self {
        Self { lr, ..Default::default() }
    }
}

/// Distributed AdamW optimizer
///
/// Implements the AdamW algorithm with explicit gradient input, suitable for
/// distributed training where gradients are synchronized before the optimizer step.
///
/// # AdamW Algorithm
///
/// For each parameter θ with gradient g:
/// ```text
/// m = β₁ * m + (1 - β₁) * g           # Update biased first moment
/// v = β₂ * v + (1 - β₂) * g²          # Update biased second moment
/// m̂ = m / (1 - β₁ᵗ)                   # Bias-corrected first moment
/// v̂ = v / (1 - β₂ᵗ)                   # Bias-corrected second moment
/// θ = θ - lr * (m̂ / (√v̂ + ε) + λ * θ) # Update with weight decay
/// ```
pub struct DistributedAdamW {
    /// First moment estimates (m)
    m: HashMap<String, Tensor>,
    /// Second moment estimates (v)
    v: HashMap<String, Tensor>,
    /// Configuration
    config: AdamWConfig,
    /// Current step (for bias correction)
    step_count: usize,
}

impl DistributedAdamW {
    /// Create a new DistributedAdamW optimizer
    ///
    /// # Arguments
    ///
    /// * `varmap` - VarMap containing all parameters (used to initialize moment buffers)
    /// * `config` - Optimizer configuration
    pub fn new(varmap: &VarMap, config: AdamWConfig) -> Result<Self> {
        let data = varmap.data();
        let mut m = HashMap::with_capacity(data.len());
        let mut v = HashMap::with_capacity(data.len());

        // Initialize moment buffers to zeros
        for (name, var) in data.iter() {
            let tensor = var.as_tensor();
            let zeros = Tensor::zeros(tensor.shape(), tensor.dtype(), tensor.device())?;
            m.insert(name.clone(), zeros.clone());
            v.insert(name.clone(), zeros);
        }

        Ok(Self {
            m,
            v,
            config,
            step_count: 0,
        })
    }

    /// Create with just a learning rate (using default config)
    #[allow(dead_code)]
    pub fn with_lr(varmap: &VarMap, lr: f64) -> Result<Self> {
        Self::new(varmap, AdamWConfig::with_lr(lr))
    }

    /// Perform an optimizer step with synchronized gradients
    ///
    /// # Arguments
    ///
    /// * `synced_grads` - Vector of (param_name, tensor, gradient) from gradient synchronizer
    /// * `varmap` - VarMap containing all parameters (for updating)
    ///
    /// # Returns
    ///
    /// Ok(()) on success, or error if shapes don't match
    pub fn step(&mut self, synced_grads: &[SyncedGradient], varmap: &mut VarMap) -> Result<()> {
        self.step_count += 1;
        let t = self.step_count as f64;

        // Bias correction terms
        let bias_correction1 = 1.0 - self.config.beta1.powf(t);
        let bias_correction2 = 1.0 - self.config.beta2.powf(t);

        for (name, param_tensor, grad) in synced_grads {
            // Get moment buffers
            let m_t = self.m.get_mut(name).ok_or_else(|| {
                Error::Msg(format!("Unknown parameter: {}", name))
            })?;
            let v_t = self.v.get_mut(name).ok_or_else(|| {
                Error::Msg(format!("Unknown parameter: {}", name))
            })?;

            // Ensure gradient is on the same device as parameter
            // Note: Device doesn't implement PartialEq, so we compare by debug representation
            let grad = {
                let grad_device = format!("{:?}", grad.device());
                let param_device = format!("{:?}", param_tensor.device());
                if grad_device != param_device {
                    grad.to_device(param_tensor.device())?
                } else {
                    grad.clone()
                }
            };

            // m = β₁ * m + (1 - β₁) * g
            let m_scaled = (m_t.clone() * self.config.beta1)?;
            let grad_scaled = (grad.clone() * (1.0 - self.config.beta1))?;
            let new_m = m_scaled.add(&grad_scaled)?;

            // v = β₂ * v + (1 - β₂) * g²
            let grad_sq = grad.sqr()?;
            let v_scaled = (v_t.clone() * self.config.beta2)?;
            let grad_sq_scaled = (grad_sq * (1.0 - self.config.beta2))?;
            let new_v = v_scaled.add(&grad_sq_scaled)?;

            // Store updated moments
            *m_t = new_m.clone();
            *v_t = new_v.clone();

            // Bias-corrected moments
            let m_hat = (&new_m / bias_correction1)?;
            let v_hat = (&new_v / bias_correction2)?;

            // Compute update: lr * (m̂ / (√v̂ + ε))
            let v_sqrt = v_hat.sqrt()?;
            let v_sqrt_eps = (&v_sqrt + self.config.eps)?;
            let adam_update = (m_hat.div(&v_sqrt_eps)? * self.config.lr)?;

            // Weight decay: lr * λ * θ
            let param = param_tensor.as_tensor();
            let weight_decay_update = (param * (self.config.lr * self.config.weight_decay))?;

            // θ = θ - adam_update - weight_decay_update
            let param_after_adam = param.sub(&adam_update)?;
            let new_param = param_after_adam.sub(&weight_decay_update)?;

            // Update the parameter in VarMap
            varmap.set(name, new_param);
        }

        Ok(())
    }

    /// Get the current step count
    #[allow(dead_code)]
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get the learning rate
    #[allow(dead_code)]
    pub fn learning_rate(&self) -> f64 {
        self.config.lr
    }

    /// Set the learning rate (for learning rate scheduling)
    #[allow(dead_code)]
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.config.lr = lr;
    }
}

/// Compute gradient norm from synchronized gradients
pub fn compute_grad_norm(synced_grads: &[SyncedGradient]) -> Result<f64> {
    let mut total_norm_sq = 0.0f64;

    for (_, _, grad) in synced_grads {
        let grad_norm_sq = grad.sqr()?
            .sum_all()?
            .to_dtype(DType::F64)?
            .to_vec0::<f64>()?;
        total_norm_sq += grad_norm_sq;
    }

    Ok(total_norm_sq.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adamw_config_default() {
        let config = AdamWConfig::default();
        assert_eq!(config.lr, 3e-4);
        assert_eq!(config.beta1, 0.9);
        assert_eq!(config.beta2, 0.999);
    }

    #[test]
    fn test_adamw_config_with_lr() {
        let config = AdamWConfig::with_lr(1e-4);
        assert_eq!(config.lr, 1e-4);
        assert_eq!(config.beta1, 0.9); // Default
    }
}
