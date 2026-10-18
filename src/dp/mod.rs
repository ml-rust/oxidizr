//! Data-parallel training: the cross-rank loss weighting, the gradient sum,
//! the rank-0 broadcasts, and the trainer hooks over them.

mod broadcast;
mod reduce;
mod trainer_hooks;
mod weighting;

pub use broadcast::{broadcast_model_parameters, broadcast_stop_flag};
pub use reduce::reduce_step_gradients;
pub use trainer_hooks::{AbortHook, PEER_STOP_REASON};
pub use weighting::reduce_step_weighting;
