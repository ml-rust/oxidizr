//! NCCL-based gradient synchronization backend
//!
//! This module provides high-performance gradient synchronization using boostr's
//! NCCL wrapper for all-reduce operations.
//!
//! # Requirements
//!
//! - CUDA 12.x or 13.x
//! - NCCL library installed on the system
//! - Compile with `--features nccl`
//!
//! # Performance
//!
//! NCCL provides optimal bandwidth utilization for GPU-to-GPU communication:
//! - NVLink: up to 600 GB/s bidirectional
//! - PCIe: ~12-32 GB/s depending on generation
//!
//! For 4+ GPUs, NCCL is significantly faster than CPU staging.

use crate::prelude::{DType, Device, Error, GradStore, Result, Tensor, VarMap};
use std::sync::Arc;

use boostr::distributed::{AllReduceOp, NcclContext};

use super::sync::{GradientSynchronizer, SyncedGradient};

/// NCCL-based gradient synchronizer
///
/// Uses boostr's NCCL wrapper for high-performance gradient synchronization
/// across multiple GPUs.
///
/// # Thread Safety
///
/// This struct wraps boostr's NcclContext which handles thread safety internally.
pub struct NcclSynchronizer {
    /// GPU devices participating in training (Candle devices)
    devices: Vec<Device>,
    /// boostr NCCL context for all-reduce operations
    nccl: Arc<NcclContext>,
    /// Number of GPUs
    world_size: usize,
}

// SAFETY: NcclContext handles thread safety internally via Mutex
unsafe impl Send for NcclSynchronizer {}
unsafe impl Sync for NcclSynchronizer {}

impl NcclSynchronizer {
    /// Create a new NCCL synchronizer for the specified GPUs
    ///
    /// # Arguments
    ///
    /// * `gpu_ids` - GPU ordinal IDs to use (e.g., [0, 1, 2, 3])
    ///
    /// # Returns
    ///
    /// New synchronizer instance or error if initialization fails
    pub fn new(gpu_ids: &[usize]) -> Result<Self> {
        let world_size = gpu_ids.len();
        if world_size < 2 {
            return Err(Error::DistributedError {
                reason: "NcclSynchronizer requires at least 2 GPUs. Use CpuStagingSynchronizer for single GPU."
                    .to_string(),
            });
        }

        // Create Candle devices
        let devices: Vec<Device> = gpu_ids
            .iter()
            .map(|&id| Device::new_cuda(id))
            .collect::<Result<Vec<_>>>()?;

        // Create boostr NCCL context
        let nccl = NcclContext::new(gpu_ids).map_err(|e| Error::DistributedError {
            reason: format!("Failed to create NCCL context: {}", e),
        })?;

        log::info!(
            "NCCL synchronizer initialized with {} GPUs: {:?} (via boostr)",
            world_size,
            gpu_ids
        );

        Ok(Self {
            devices,
            nccl: Arc::new(nccl),
            world_size,
        })
    }

    /// Get the primary device (GPU 0)
    #[allow(dead_code)]
    pub fn primary_device(&self) -> &Device {
        &self.devices[0]
    }

    /// Perform NCCL all-reduce on a tensor
    ///
    /// This uses boostr's CPU-staged all-reduce for Candle tensor interop.
    fn all_reduce_tensor(&self, tensor: &Tensor) -> Result<Tensor> {
        let dtype = tensor.dtype();

        // Flatten tensor for all-reduce
        let flat = tensor.flatten_all()?;

        // Get raw data and perform all-reduce based on dtype
        match dtype {
            DType::F32 => {
                let data: Vec<f32> = flat.to_vec1()?;
                let reduced = self.all_reduce_f32(&data)?;
                Tensor::from_vec(reduced, tensor.shape(), tensor.device())
            }
            DType::F16 => {
                // Convert to f32 for all-reduce, then back
                let f32_tensor = flat.to_dtype(DType::F32)?;
                let data: Vec<f32> = f32_tensor.to_vec1()?;
                let reduced = self.all_reduce_f32(&data)?;
                let result = Tensor::from_vec(reduced, tensor.shape(), tensor.device())?;
                result.to_dtype(DType::F16)
            }
            DType::BF16 => {
                // Convert to f32 for all-reduce, then back
                let f32_tensor = flat.to_dtype(DType::F32)?;
                let data: Vec<f32> = f32_tensor.to_vec1()?;
                let reduced = self.all_reduce_f32(&data)?;
                let result = Tensor::from_vec(reduced, tensor.shape(), tensor.device())?;
                result.to_dtype(DType::BF16)
            }
            _ => Err(Error::DistributedError {
                reason: format!("NCCL all-reduce not supported for dtype {:?}", dtype),
            }),
        }
    }

    /// All-reduce f32 data using boostr's NCCL
    fn all_reduce_f32(&self, data: &[f32]) -> Result<Vec<f32>> {
        self.nccl
            .all_reduce_cpu_staged_f32(data, AllReduceOp::Avg)
            .map_err(|e| Error::DistributedError {
                reason: format!("NCCL all_reduce failed: {}", e),
            })
    }
}

impl GradientSynchronizer for NcclSynchronizer {
    fn synchronize_gradients(
        &self,
        grads: &GradStore,
        varmap: &VarMap,
    ) -> Result<Vec<SyncedGradient>> {
        let data = varmap.data().lock().unwrap();
        let mut result = Vec::with_capacity(data.len());

        for (name, var) in data.iter() {
            if let Some(grad) = grads.get(var) {
                // Perform NCCL all-reduce to average gradients
                let synced_grad = self.all_reduce_tensor(&grad)?;
                result.push((name.clone(), var.clone(), synced_grad));
            }
        }

        Ok(result)
    }

    fn local_rank(&self) -> usize {
        0 // Primary GPU in single-process mode
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn barrier(&self) -> Result<()> {
        self.nccl.barrier().map_err(|e| Error::DistributedError {
            reason: format!("Barrier failed: {}", e),
        })
    }
}
