//! Core trait and types for gradient synchronization
//!
//! This module defines the `GradientSynchronizer` trait that all backends must implement,
//! as well as configuration types for distributed training.

use crate::prelude::{GradStore, Result, Tensor, VarMap};
use serde::{Deserialize, Serialize};

/// Synchronization backend selection
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SyncBackend {
    /// No synchronization (single GPU mode)
    #[default]
    Single,
    /// CPU-based gradient staging and averaging
    #[serde(alias = "cpu")]
    CpuStaging,
    /// NCCL-based all-reduce (requires 'nccl' feature)
    Nccl,
}

impl std::str::FromStr for SyncBackend {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "single" | "none" => Ok(SyncBackend::Single),
            "cpu" | "cpu_staging" | "cpustaging" => Ok(SyncBackend::CpuStaging),
            "nccl" => Ok(SyncBackend::Nccl),
            _ => Err(format!("Unknown sync backend: {}. Valid options: single, cpu, nccl", s)),
        }
    }
}

/// Configuration for distributed training
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct DistributedConfig {
    /// Enable distributed training
    #[serde(default)]
    pub enabled: bool,

    /// Synchronization backend
    #[serde(default)]
    pub backend: SyncBackend,

    /// GPU IDs to use (empty = auto-detect all available)
    #[serde(default)]
    pub gpu_ids: Vec<usize>,
}

#[allow(dead_code)]
impl DistributedConfig {
    /// Create a single-GPU configuration (no synchronization)
    pub fn single_gpu(gpu_id: usize) -> Self {
        Self {
            enabled: false,
            backend: SyncBackend::Single,
            gpu_ids: vec![gpu_id],
        }
    }

    /// Create a multi-GPU configuration
    pub fn multi_gpu(gpu_ids: Vec<usize>, backend: SyncBackend) -> Self {
        Self {
            enabled: gpu_ids.len() > 1,
            backend,
            gpu_ids,
        }
    }

    /// Get the number of GPUs
    pub fn world_size(&self) -> usize {
        self.gpu_ids.len().max(1)
    }

    /// Get the effective batch size multiplier
    pub fn effective_batch_multiplier(&self) -> usize {
        if self.enabled {
            self.gpu_ids.len().max(1)
        } else {
            1
        }
    }
}

/// Synchronized gradient: parameter name, parameter tensor, and its averaged gradient tensor
pub type SyncedGradient = (String, Tensor, Tensor);

/// Trait for gradient synchronization across multiple GPUs
///
/// Implementations of this trait handle the communication between GPUs
/// to average gradients during distributed data parallel training.
///
/// # Thread Safety
///
/// This trait requires `Send + Sync` because it may be used across threads
/// in the training loop.
#[allow(dead_code)]
pub trait GradientSynchronizer: Send + Sync {
    /// Synchronize gradients across all GPUs
    ///
    /// Takes the gradient store from the backward pass and the VarMap containing
    /// all model parameters. Returns a list of (param_name, var, averaged_gradient)
    /// tuples that can be used to update the model.
    ///
    /// # Arguments
    ///
    /// * `grads` - GradStore from `loss.backward()`
    /// * `varmap` - VarMap containing all model parameters
    ///
    /// # Returns
    ///
    /// Vector of (parameter_name, Var, averaged_gradient) tuples
    fn synchronize_gradients(
        &self,
        grads: &GradStore,
        varmap: &VarMap,
    ) -> Result<Vec<SyncedGradient>>;

    /// Get the local rank (GPU index within this process)
    fn local_rank(&self) -> usize;

    /// Get total number of GPUs participating
    fn world_size(&self) -> usize;

    /// Check if this is the primary GPU (rank 0)
    fn is_primary(&self) -> bool {
        self.local_rank() == 0
    }

    /// Synchronization barrier - wait for all GPUs
    fn barrier(&self) -> Result<()>;

    /// Check if distributed training is actually active
    fn is_distributed(&self) -> bool {
        self.world_size() > 1
    }
}
