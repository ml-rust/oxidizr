//! Data-parallel training: the cross-rank loss weighting, the gradient sum,
//! the rank-0 broadcasts, the trainer hooks over them, the resume checks,
//! and the in-process launcher with each rank's device.

mod broadcast;
mod device;
mod launch;
mod reduce;
mod resume;
mod trainer_hooks;
mod weighting;

pub use broadcast::{broadcast_model_parameters, broadcast_stop_flag};
pub use device::RankDevice;
pub use launch::{run_ranks, RankGroup, RankSlot};
pub use reduce::reduce_step_gradients;
pub use resume::check_resume_world;
pub use trainer_hooks::{AbortHook, PEER_STOP_REASON};
pub use weighting::reduce_step_weighting;
