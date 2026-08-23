//! CLI subcommands for oxidizr
//!
//! This module contains the implementation of all CLI subcommands:
//! - `train` - Train a model (default command)
//! - `import` - Turn a HuggingFace checkpoint into an oxidizr checkpoint
//! - `merge` - Fold LoRA adapters into their base weights
//! - `pack` - Package a trained model for distribution
//! - `push` - Push a packaged model to HuggingFace (feature-gated)

pub mod import;
pub mod merge;
pub mod pack;
pub mod train;

#[cfg(feature = "huggingface")]
pub mod push;

pub use import::{run_import, ImportArgs};
pub use merge::{run_merge, MergeArgs};
pub use pack::{run_pack, PackArgs};
pub use train::{run_train, TrainArgs};

#[cfg(feature = "huggingface")]
pub use push::{run_push, PushArgs};
