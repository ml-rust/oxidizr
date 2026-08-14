//! CLI subcommands for oxidizr
//!
//! This module contains the implementation of all CLI subcommands:
//! - `train` - Train a model (default command)
//! - `pack` - Package a trained model for distribution
//! - `push` - Push a packaged model to HuggingFace (feature-gated)

pub mod pack;
pub mod train;

#[cfg(feature = "huggingface")]
pub mod push;

pub use pack::{run_pack, PackArgs};
pub use train::{run_train, TrainArgs};

#[cfg(feature = "huggingface")]
pub use push::{run_push, PushArgs};
