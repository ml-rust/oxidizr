//! Training metrics: token throughput, FLOPs and MFU, GPU memory, TensorBoard
//! events, and the loss-spike guard.

mod config;
mod flops;
mod nvml;
mod scalars;
mod spike;
mod state;
#[cfg(feature = "tensorboard")]
mod tensorboard;
mod trainer_metrics;

pub use config::{MetricsConfig, SpikeAction, SpikeGuardConfig};
pub use flops::{flops_per_token, mfu, model_flops_per_token, FlopsModel, MatmulParam};
pub use nvml::{device_mib_text, gpu_memory, process_mib_text, summary_text, GpuMemory};
pub use scalars::{tags, ScalarLog};
pub use spike::{read_spike_guard_state, SpikeEvent, SpikeGuard, SpikeGuardState};
pub use state::{LossProbe, RunMetrics, TrainMetrics};
#[cfg(feature = "tensorboard")]
pub use tensorboard::{
    crc32c, encode_event, frame_record, masked_crc32c, EventBody, EventWriter, FILE_VERSION,
};
