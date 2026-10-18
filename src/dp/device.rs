//! The device of each rank, and the communicator group over those devices.
//!
//! The numr `Runtime` trait has no device by index, so each runtime of a
//! training run implements [`RankDevice`].

use super::launch::RankGroup;
use crate::prelude::DType;
use anyhow::{bail, Result};
use boostr::runtime::Runtime;

/// Selects a rank's device from the `--gpus` list and joins ranks into a group.
pub trait RankDevice: Runtime<DType = DType> {
    /// The device and client of the rank on GPU `gpu_id`.
    ///
    /// `None` is the runtime's default device, as a run without `--gpus` uses.
    ///
    /// # Errors
    /// Returns an error when the device does not open.
    fn rank_device(gpu_id: Option<usize>) -> Result<(Self::Device, Self::Client)>;

    /// The communicator group of one rank per client, rank `i` on `clients[i]`.
    ///
    /// # Errors
    /// Returns an error when this runtime or build cannot run `clients.len()` ranks.
    fn rank_group(clients: &[Self::Client]) -> Result<RankGroup>;
}

impl RankDevice for boostr::CpuRuntime {
    /// The CPU device for every rank. A single GPU id is ignored.
    fn rank_device(_gpu_id: Option<usize>) -> Result<(Self::Device, Self::Client)> {
        let device = Self::default_device();
        let client = Self::default_client(&device);
        Ok((device, client))
    }

    fn rank_group(clients: &[Self::Client]) -> Result<RankGroup> {
        match clients.len() {
            1 => Ok(RankGroup::single()),
            world => bail!(
                "the CPU runtime cannot run {world} ranks from `--gpus`: it has no GPUs. \
                 Multi-GPU training needs `--target-device gpu` and a build with \
                 `--features cuda,nccl`"
            ),
        }
    }
}

#[cfg(feature = "cuda")]
impl RankDevice for boostr::CudaRuntime {
    fn rank_device(gpu_id: Option<usize>) -> Result<(Self::Device, Self::Client)> {
        let Some(id) = gpu_id else {
            let device = Self::default_device();
            let client = Self::default_client(&device);
            return Ok((device, client));
        };
        let device = boostr::CudaDevice::new(id);
        // A client is cached per device index, so every tensor of this rank
        // allocates through this client's stream.
        let client = boostr::CudaClient::new(device.clone()).map_err(|e| {
            anyhow::anyhow!(
                "GPU {id} did not open: {e:?}. Check that `--gpus` names a visible GPU \
                 (see `nvidia-smi`)"
            )
        })?;
        Ok((device, client))
    }

    fn rank_group(clients: &[Self::Client]) -> Result<RankGroup> {
        match clients.len() {
            1 => Ok(RankGroup::single()),
            #[cfg(feature = "nccl")]
            _ => RankGroup::nccl(clients),
            #[cfg(not(feature = "nccl"))]
            world => bail!(
                "{world} GPUs need NCCL, and this build has no NCCL support. Rebuild with \
                 `--features cuda,nccl`, or pass one GPU id"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::CpuRuntime;

    #[test]
    fn the_cpu_runs_one_rank_and_refuses_more() {
        let (_, client) = CpuRuntime::rank_device(Some(0)).expect("cpu device");
        assert_eq!(
            CpuRuntime::rank_group(std::slice::from_ref(&client))
                .expect("one rank")
                .world_size(),
            1
        );
        let Err(err) = CpuRuntime::rank_group(&[client.clone(), client]) else {
            panic!("two CPU ranks must be refused");
        };
        assert!(err.to_string().contains("--features cuda,nccl"), "{err}");
    }
}
