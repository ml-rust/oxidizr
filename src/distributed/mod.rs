//! Multi-GPU Data Parallelism Module
//!
//! This module provides distributed training support for oxidizr, enabling
//! training across multiple GPUs on a single machine.
//!
//! # Architecture
//!
//! The module uses a trait-based design with pluggable backends:
//!
//! - `GradientSynchronizer` - Core trait for gradient synchronization
//! - `CpuStagingSynchronizer` - Default backend using CPU for gradient averaging
//! - `NcclSynchronizer` - High-performance NCCL backend (feature-gated)
//!
//! # Usage
//!
//! ```ignore
//! // Create synchronizer based on config
//! let sync = CpuStagingSynchronizer::new(&[0, 1, 2, 3])?;
//!
//! // In training loop, after backward pass:
//! let synced_grads = sync.synchronize_gradients(&grads, &varmap)?;
//! optimizer.step(&synced_grads)?;
//! ```

pub mod cpu_staging;
pub mod optimizer;
pub mod sync;

#[cfg(feature = "nccl")]
pub mod nccl;

// Re-exports for convenience
pub use cpu_staging::{CpuStagingSynchronizer, RuntimeDeviceFromOrdinal};
pub use optimizer::{AdamWConfig, DistributedAdamW};
pub use sync::{DistributedConfig, GradientSynchronizer, SyncBackend};

#[cfg(feature = "nccl")]
pub use nccl::NcclSynchronizer;

use crate::prelude::{Error, Result};

/// Create a synchronizer based on configuration
pub fn create_synchronizer<R>(
    config: &DistributedConfig,
) -> Result<Box<dyn GradientSynchronizer<R>>>
where
    R: RuntimeDeviceFromOrdinal,
{
    if !config.enabled || config.gpu_ids.len() <= 1 {
        // Single GPU - use no-op synchronizer
        return Ok(Box::new(CpuStagingSynchronizer::single_gpu(
            config.gpu_ids.first().copied().unwrap_or(0),
        )?));
    }

    match config.backend {
        SyncBackend::Single => Ok(Box::new(CpuStagingSynchronizer::single_gpu(
            config.gpu_ids[0],
        )?)),
        SyncBackend::CpuStaging => Ok(Box::new(CpuStagingSynchronizer::new(&config.gpu_ids)?)),
        #[cfg(feature = "nccl")]
        SyncBackend::Nccl => Ok(Box::new(NcclSynchronizer::new(&config.gpu_ids)?)),
        #[cfg(not(feature = "nccl"))]
        SyncBackend::Nccl => Err(Error::DistributedError {
            reason: "NCCL backend requested but 'nccl' feature is not enabled. \
                 Compile with --features nccl to enable NCCL support."
                .to_string(),
        }),
    }
}
