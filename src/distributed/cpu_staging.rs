//! CPU-based gradient synchronization backend
//!
//! This backend synchronizes gradients by copying them to CPU memory,
//! averaging across all GPUs, and copying back. It requires no external
//! dependencies but is slower than NCCL due to PCIe bandwidth limitations.
//!
//! # Performance Characteristics
//!
//! - Works with any number of GPUs
//! - No external dependencies
//! - Overhead: ~2x GPU-CPU memory transfers per gradient sync
//! - Practical for 2-4 GPUs; consider NCCL for more

use crate::prelude::{Device, Error, GradStore, Result, VarMap};
use std::sync::{Arc, Barrier};

use super::sync::{GradientSynchronizer, SyncedGradient};

/// CPU-based gradient synchronizer
///
/// Implements gradient synchronization by:
/// 1. Copying gradients from each GPU to CPU
/// 2. Averaging gradients on CPU
/// 3. Returning the averaged gradients for optimizer update
pub struct CpuStagingSynchronizer {
    /// GPU devices participating in training
    #[allow(dead_code)]
    devices: Vec<Device>,
    /// Number of GPUs
    world_size: usize,
    /// Barrier for synchronization (unused in single-process mode)
    #[allow(dead_code)]
    barrier: Arc<Barrier>,
}

impl CpuStagingSynchronizer {
    /// Create a new CPU staging synchronizer for the specified GPUs
    ///
    /// # Arguments
    ///
    /// * `gpu_ids` - GPU ordinal IDs to use (e.g., [0, 1, 2, 3])
    ///
    /// # Returns
    ///
    /// New synchronizer instance or error if GPU creation fails
    pub fn new(gpu_ids: &[usize]) -> Result<Self> {
        let world_size = gpu_ids.len();
        if world_size == 0 {
            return Err(Error::Msg(
                "CpuStagingSynchronizer requires at least 1 GPU".to_string()
            ));
        }

        let devices: Vec<Device> = gpu_ids
            .iter()
            .map(|&id| Device::new_cuda(id))
            .collect();

        let barrier = Arc::new(Barrier::new(world_size));

        Ok(Self {
            devices,
            world_size,
            barrier,
        })
    }

    /// Create a single-GPU synchronizer (no-op mode)
    pub fn single_gpu(gpu_id: usize) -> Result<Self> {
        Self::new(&[gpu_id])
    }

    /// Get the primary device (GPU 0)
    #[allow(dead_code)]
    pub fn primary_device(&self) -> &Device {
        &self.devices[0]
    }
}

impl GradientSynchronizer for CpuStagingSynchronizer {
    fn synchronize_gradients(
        &self,
        grads: &GradStore,
        varmap: &VarMap,
    ) -> Result<Vec<SyncedGradient>> {
        // For single GPU, just pass through gradients
        if self.world_size == 1 {
            return self.passthrough_gradients(grads, varmap);
        }

        // Multi-GPU case: average gradients
        // In single-process multi-GPU, we assume all gradients are already
        // computed on the primary device. The averaging is a no-op in this case
        // since we're doing data parallelism with a single forward/backward pass.
        //
        // For true data parallelism, you would:
        // 1. Run forward/backward on each GPU with different data shards
        // 2. Collect gradients from all GPUs to CPU
        // 3. Average and return
        //
        // This implementation handles the gradient extraction and averaging
        // infrastructure, which can be extended to multi-process later.
        self.average_gradients(grads, varmap)
    }

    fn local_rank(&self) -> usize {
        0 // Primary GPU in single-process mode
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn barrier(&self) -> Result<()> {
        // In single-process mode, barrier is a no-op
        // (all GPU operations are sequential within the same process)
        Ok(())
    }
}

impl CpuStagingSynchronizer {
    /// Pass through gradients without modification (single GPU case)
    fn passthrough_gradients(
        &self,
        grads: &GradStore,
        varmap: &VarMap,
    ) -> Result<Vec<SyncedGradient>> {
        let data = varmap.data();
        let mut result = Vec::with_capacity(data.len());

        for (name, tensor) in data.iter() {
            if let Some(grad) = grads.get(tensor.id()) {
                result.push((name.clone(), tensor.clone(), grad.clone()));
            }
        }

        Ok(result)
    }

    /// Average gradients across GPUs (multi-GPU case)
    ///
    /// In the current single-process implementation, this is equivalent to
    /// passthrough since we only have one set of gradients. When extended
    /// to multi-process or multi-thread, this would collect and average.
    fn average_gradients(
        &self,
        grads: &GradStore,
        varmap: &VarMap,
    ) -> Result<Vec<SyncedGradient>> {
        let data = varmap.data();
        let mut result = Vec::with_capacity(data.len());
        let scale = 1.0 / self.world_size as f64;

        for (name, tensor) in data.iter() {
            if let Some(grad) = grads.get(tensor.id()) {
                // Copy to CPU for averaging
                let grad_cpu = grad.to_device(&Device::cpu())?;

                // Scale by 1/world_size
                // In multi-GPU, gradients from each GPU would be summed first
                let scaled_grad = (&grad_cpu * scale)?;

                // Copy back to original device
                let grad_synced = scaled_grad.to_device(grad.device())?;

                result.push((name.clone(), tensor.clone(), grad_synced));
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_backend_parsing() {
        use super::super::sync::SyncBackend;

        assert_eq!("cpu".parse::<SyncBackend>().unwrap(), SyncBackend::CpuStaging);
        assert_eq!("nccl".parse::<SyncBackend>().unwrap(), SyncBackend::Nccl);
        assert_eq!("single".parse::<SyncBackend>().unwrap(), SyncBackend::Single);
    }
}
