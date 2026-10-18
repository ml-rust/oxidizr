//! Distributed config: the sync backend, the GPU id list, and the multi-GPU checks.

pub mod gpus;
pub mod sync;

pub use gpus::{
    check_gpu_ids, check_sync_backend, parse_gpu_ids, reject_duplicate_gpus, BuildSupport,
};
pub use sync::{DistributedConfig, SyncBackend};
