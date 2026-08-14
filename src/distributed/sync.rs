//! Core trait and types for gradient synchronization
//!
//! This module defines the `GradientSynchronizer` trait that all backends must implement,
//! as well as configuration types for distributed training.

use crate::prelude::{DType, Result, Tensor};
use boostr::autograd::GradStore;
use boostr::runtime::Runtime;
use boostr::tensor::TensorId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
            _ => Err(format!(
                "Unknown sync backend: {}. Valid options: single, cpu, nccl",
                s
            )),
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

/// Synchronized gradient entry keyed by the stable parameter `TensorId`.
#[allow(dead_code)]
pub type SyncedGradient<R> = (TensorId, Tensor<R>);

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
pub trait GradientSynchronizer<R>: Send + Sync
where
    R: Runtime<DType = DType>,
{
    /// Synchronize gradients across all GPUs.
    ///
    /// Takes the gradient store from the backward pass and the optimizer-owned
    /// parameter map. Implementations must return gradients keyed by the same
    /// stable `TensorId`s from `params`; changing keys would orphan AdamW state.
    ///
    /// # Arguments
    ///
    /// * `grads` - GradStore from `backward()`
    /// * `params` - Optimizer-owned parameter map keyed by stable TensorId
    ///
    /// # Returns
    ///
    /// A `GradStore` containing averaged gradients under the original TensorId keys.
    fn synchronize_gradients(
        &self,
        grads: &GradStore<R>,
        params: &HashMap<TensorId, Tensor<R>>,
    ) -> Result<GradStore<R>>;

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
