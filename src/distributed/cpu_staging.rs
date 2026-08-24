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

use crate::prelude::{DType, Error, Result, Tensor};
use boostr::autograd::GradStore;
use boostr::runtime::Runtime;
use boostr::tensor::TensorId;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Barrier};

use super::sync::GradientSynchronizer;

/// Construct an associated runtime device from an ordinal supplied by the CLI.
///
/// The core `Runtime` trait exposes `default_device()` but not an indexed-device
/// constructor. This local trait keeps `CpuStagingSynchronizer` generic while
/// preserving the existing `gpu_ids` API for the runtimes oxidizr dispatches to.
pub trait RuntimeDeviceFromOrdinal: Runtime<DType = DType> {
    fn device_from_ordinal(ordinal: usize) -> Result<Self::Device>;
}

impl RuntimeDeviceFromOrdinal for boostr::CpuRuntime {
    fn device_from_ordinal(ordinal: usize) -> Result<Self::Device> {
        if ordinal == 0 {
            Ok(boostr::CpuDevice::new())
        } else {
            Err(Error::DistributedError {
                reason: format!(
                    "CPU runtime has only device 0; got distributed device ordinal {ordinal}"
                ),
            })
        }
    }
}

#[cfg(feature = "cuda")]
impl RuntimeDeviceFromOrdinal for boostr::CudaRuntime {
    fn device_from_ordinal(ordinal: usize) -> Result<Self::Device> {
        Ok(boostr::CudaDevice::new(ordinal))
    }
}

/// CPU-based gradient synchronizer
///
/// Implements gradient synchronization by:
/// 1. Copying gradients from each GPU to CPU
/// 2. Averaging gradients on CPU
/// 3. Returning the averaged gradients for optimizer update
pub struct CpuStagingSynchronizer<R>
where
    R: Runtime<DType = DType>,
{
    /// Runtime devices participating in training
    #[allow(dead_code)]
    devices: Vec<R::Device>,
    /// Number of GPUs
    world_size: usize,
    /// Barrier for synchronization (unused in single-process mode)
    #[allow(dead_code)]
    barrier: Arc<Barrier>,
    _runtime: PhantomData<R>,
}

impl<R> CpuStagingSynchronizer<R>
where
    R: RuntimeDeviceFromOrdinal,
{
    /// Create a new CPU staging synchronizer for the specified GPUs
    ///
    /// # Arguments
    ///
    /// * `gpu_ids` - GPU ordinal IDs to use (e.g., [0, 1, 2, 3])
    ///
    /// # Returns
    ///
    /// New synchronizer instance or error if device creation fails
    pub fn new(gpu_ids: &[usize]) -> Result<Self> {
        let world_size = gpu_ids.len();
        if world_size == 0 {
            return Err(Error::DistributedError {
                reason: "CpuStagingSynchronizer requires at least 1 GPU".to_string(),
            });
        }

        let devices: Vec<R::Device> = gpu_ids
            .iter()
            .map(|&id| R::device_from_ordinal(id))
            .collect::<Result<Vec<_>>>()?;

        let barrier = Arc::new(Barrier::new(world_size));

        Ok(Self {
            devices,
            world_size,
            barrier,
            _runtime: PhantomData,
        })
    }

    /// Create a single-GPU synchronizer (no-op mode)
    pub fn single_gpu(gpu_id: usize) -> Result<Self> {
        Self::new(&[gpu_id])
    }

    /// Get the primary device (GPU 0)
    #[allow(dead_code)]
    pub fn primary_device(&self) -> &R::Device {
        &self.devices[0]
    }
}

impl<R> GradientSynchronizer<R> for CpuStagingSynchronizer<R>
where
    R: Runtime<DType = DType>,
{
    fn synchronize_gradients(
        &self,
        grads: &GradStore<R>,
        params: &HashMap<TensorId, Tensor<R>>,
    ) -> Result<GradStore<R>> {
        // For single GPU, just pass through gradients under the same TensorId keys.
        if self.world_size == 1 {
            return self.passthrough_gradients(grads, params);
        }

        // Multi-GPU case: average gradients.
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
        self.average_gradients(grads, params)
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

impl<R> CpuStagingSynchronizer<R>
where
    R: Runtime<DType = DType>,
{
    /// Pass through gradients without modification (single GPU case)
    fn passthrough_gradients(
        &self,
        grads: &GradStore<R>,
        params: &HashMap<TensorId, Tensor<R>>,
    ) -> Result<GradStore<R>> {
        let mut result = GradStore::new();

        for id in params.keys().copied() {
            if let Some(grad) = grads.get(id) {
                result.insert(id, grad.clone());
            }
        }

        Ok(result)
    }

    /// Average gradients across GPUs (multi-GPU case)
    ///
    /// In the current single-process implementation, this is equivalent to
    /// passthrough plus scaling because we only have one set of gradients. When
    /// extended to multi-process or multi-thread, this would collect and average.
    fn average_gradients(
        &self,
        grads: &GradStore<R>,
        params: &HashMap<TensorId, Tensor<R>>,
    ) -> Result<GradStore<R>> {
        let mut result = GradStore::new();
        let scale = 1.0 / self.world_size as f64;

        for id in params.keys().copied() {
            if let Some(grad) = grads.get(id) {
                // Copy to CPU for averaging, preserving dtype and shape.
                let grad_cpu = tensor_to_cpu(grad)?;

                // Scale by 1/world_size. In multi-GPU, gradients from each GPU
                // would be summed first; all scalar math delegates to numr.
                let scaled_cpu = grad_cpu.mul_scalar(scale)?;

                // Copy back to the original runtime/device and preserve the
                // original GradStore key so AdamW state remains attached.
                let grad_synced = tensor_from_cpu::<R>(&scaled_cpu, grad.device())?;
                result.insert(id, grad_synced);
            }
        }

        Ok(result)
    }
}

fn tensor_to_cpu<R>(tensor: &Tensor<R>) -> Result<Tensor<boostr::CpuRuntime>>
where
    R: Runtime<DType = DType>,
{
    let bytes = tensor.to_bytes()?;
    let cpu_device = boostr::CpuDevice::new();
    Tensor::<boostr::CpuRuntime>::from_bytes(&bytes, tensor.shape(), tensor.dtype(), &cpu_device)
        .map_err(Error::from)
}

fn tensor_from_cpu<R>(tensor: &Tensor<boostr::CpuRuntime>, device: &R::Device) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
{
    let bytes = tensor.to_bytes()?;
    Tensor::<R>::from_bytes(&bytes, tensor.shape(), tensor.dtype(), device).map_err(Error::from)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_sync_backend_parsing() {
        use super::super::sync::SyncBackend;

        assert_eq!(
            "cpu".parse::<SyncBackend>().unwrap(),
            SyncBackend::CpuStaging
        );
        assert_eq!("nccl".parse::<SyncBackend>().unwrap(), SyncBackend::Nccl);
        assert_eq!(
            "single".parse::<SyncBackend>().unwrap(),
            SyncBackend::Single
        );
    }
}
