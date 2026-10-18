//! Distributed training config. Only one GPU is supported.

pub mod sync;

pub use sync::{reject_multi_gpu, DistributedConfig, SyncBackend};
