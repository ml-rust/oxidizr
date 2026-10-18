//! Distributed config types and the multi-GPU check.
//!
//! One GPU is the only supported world size. Any config or CLI input that
//! names more than one GPU is an error.

use crate::prelude::{Error, Result};
use serde::{Deserialize, Serialize};

/// Gradient synchronization backend name, as parsed from config and CLI.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SyncBackend {
    /// No synchronization (one GPU)
    #[default]
    Single,
    /// CPU gradient staging
    #[serde(alias = "cpu")]
    CpuStaging,
    /// NCCL all-reduce
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

/// Distributed training config. Only one GPU is supported.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct DistributedConfig {
    /// Enable distributed training
    #[serde(default)]
    pub enabled: bool,

    /// Synchronization backend
    #[serde(default)]
    pub backend: SyncBackend,

    /// GPU IDs to use
    #[serde(default)]
    pub gpu_ids: Vec<usize>,
}

impl DistributedConfig {
    /// Number of GPUs named by `gpu_ids`. An empty list counts as one.
    pub fn world_size(&self) -> usize {
        self.gpu_ids.len().max(1)
    }

    /// Returns an error when `gpu_ids` names more than one GPU.
    pub fn validate(&self) -> Result<()> {
        reject_multi_gpu("trainer.distributed.gpu_ids", &self.gpu_ids)
    }
}

/// Returns an error when `gpu_ids` names more than one GPU.
///
/// `input` is the exact user input to quote, for example `--gpus 0,1`.
pub fn reject_multi_gpu(input: &str, gpu_ids: &[usize]) -> Result<()> {
    if gpu_ids.len() <= 1 {
        return Ok(());
    }
    Err(Error::DistributedError {
        reason: format!(
            "`{input}` names {} GPUs. Multi-GPU data parallelism is not implemented in this build. \
             Pass a single GPU, for example `--gpus 0`.",
            gpu_ids.len()
        ),
    })
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
    fn world_size_above_one_is_an_error_naming_the_input() {
        let err = reject_multi_gpu("--gpus 0,1", &[0, 1]).expect_err("two GPUs");
        let text = err.to_string();
        assert!(text.contains("`--gpus 0,1`"), "{text}");
        assert!(text.contains("not implemented in this build"), "{text}");
        assert!(text.contains("single GPU"), "{text}");
    }

    #[test]
    fn config_with_two_gpu_ids_fails_validation() {
        let config = DistributedConfig {
            enabled: true,
            backend: SyncBackend::Nccl,
            gpu_ids: vec![0, 1],
        };
        assert_eq!(config.world_size(), 2);
        assert!(config.validate().is_err());
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
