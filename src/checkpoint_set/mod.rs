//! A checkpoint set: the weights, the optimizer state, and the meta that commits them.
//!
//! A trainer save writes the optimizer state, then the weights, then the meta.
//! A set without its meta is incomplete, and `--resume auto` skips it.

mod cleanup;
mod files;
mod names;
mod optimizer_io;
mod restore;
mod write;

pub use cleanup::remove_old_checkpoints;
pub use files::{optimizer_path_for, weights_step, CheckpointLabel, OPTIMIZER_SUFFIX};
pub use names::NameMaps;
pub use optimizer_io::{load_optimizer_file, read_optimizer_file, write_optimizer_file};
pub use restore::{
    apply_optimizer_resume, manual_resume_target, plan_optimizer_resume, OptimizerResume,
};
pub use write::{write_checkpoint_set, CheckpointSave};
