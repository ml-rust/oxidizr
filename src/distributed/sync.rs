//! Distributed config types: the sync backend and the GPU id list.

use super::gpus::reject_duplicate_gpus;
use crate::prelude::Result;
use serde::{Deserialize, Serialize};

/// Gradient synchronization backend name, as parsed from config and CLI.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SyncBackend {
    /// No synchronization (one GPU). The default: more than one GPU runs NCCL.
    #[default]
    Single,
    /// CPU gradient staging. No multi-rank implementation.
    #[serde(alias = "cpu")]
    CpuStaging,
    /// NCCL all-reduce, the only multi-rank backend.
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

/// Distributed training config.
///
/// `gpu_ids` names one GPU per data-parallel rank. `--gpus` overrides it.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct DistributedConfig {
    /// Enable distributed training
    #[serde(default)]
    pub enabled: bool,

    /// Synchronization backend
    #[serde(default)]
    pub backend: SyncBackend,

    /// GPU IDs to use, one per rank
    #[serde(default)]
    pub gpu_ids: Vec<usize>,
}

impl DistributedConfig {
    /// Number of GPUs named by `gpu_ids`. An empty list counts as one.
    pub fn world_size(&self) -> usize {
        self.gpu_ids.len().max(1)
    }

    /// Returns an error when `gpu_ids` names one GPU twice.
    ///
    /// Whether this build can run more than one GPU is checked by
    /// [`check_gpu_ids`](super::gpus::check_gpu_ids) at launch.
    pub fn validate(&self) -> Result<()> {
        reject_duplicate_gpus("trainer.distributed.gpu_ids", &self.gpu_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_backend_parses_known_names() {
        assert_eq!("cpu".parse::<SyncBackend>(), Ok(SyncBackend::CpuStaging));
        assert_eq!("nccl".parse::<SyncBackend>(), Ok(SyncBackend::Nccl));
        assert_eq!("single".parse::<SyncBackend>(), Ok(SyncBackend::Single));
        assert!("mpi".parse::<SyncBackend>().is_err());
    }

    #[test]
    fn config_with_two_gpu_ids_passes_validation() {
        let config = DistributedConfig {
            enabled: true,
            backend: SyncBackend::Nccl,
            gpu_ids: vec![0, 1],
        };
        assert_eq!(config.world_size(), 2);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_with_a_duplicate_gpu_id_fails_validation() {
        let config = DistributedConfig {
            gpu_ids: vec![1, 0, 1],
            ..Default::default()
        };
        let err = config.validate().expect_err("duplicate");
        assert!(err.to_string().contains("GPU 1 twice"), "{err}");
    }

    #[test]
    fn zero_or_one_gpu_passes_validation() {
        assert!(DistributedConfig::default().validate().is_ok());
        assert_eq!(DistributedConfig::default().world_size(), 1);
        let one = DistributedConfig {
            gpu_ids: vec![3],
            ..Default::default()
        };
        assert!(one.validate().is_ok());
    }
}
