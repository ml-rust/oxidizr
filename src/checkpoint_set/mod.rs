//! A checkpoint set: the weights, the optimizer state, and the meta that commits them.
//!
//! A save copies the set to host memory first, then commits it: the optimizer
//! state, then the weights, then the meta. A set without its meta is
//! incomplete, and `--resume auto` skips it. The commit runs on the training
//! thread, or on the background [`CheckpointWriter`].

mod cleanup;
mod files;
mod job;
mod names;
mod optimizer_io;
mod restore;
mod snapshot;
mod write;
mod writer;

pub use cleanup::remove_old_checkpoints;
pub use files::{optimizer_path_for, weights_step, CheckpointLabel, OPTIMIZER_SUFFIX};
pub use job::{CheckpointOutcome, Committer, Job};
pub use names::NameMaps;
pub use optimizer_io::{
    load_optimizer_file, read_optimizer_file, stage_optimizer_state, write_staged_optimizer_file,
};
pub use restore::{
    apply_optimizer_resume, manual_resume_target, plan_optimizer_resume, OptimizerResume,
};
pub use snapshot::{snapshot_set, FrozenCache, HostTensors, SetSnapshot, SnapshotInput};
pub use write::commit_set;
pub use writer::CheckpointWriter;
