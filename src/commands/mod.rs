//! CLI subcommands for oxidizr
//!
//! This module contains the implementation of all CLI subcommands:
//! - `train` - Train a model (default command)
//! - `import` - Turn a HuggingFace checkpoint into an oxidizr checkpoint
//! - `merge` - Fold LoRA adapters into their base weights
//! - `pack` - Package a trained model for distribution
//! - `prep-corpus` - Turn audio recordings into a packed speech-LM corpus
//! - `push` - Push a packaged model to HuggingFace (feature-gated)

pub mod import;
pub mod merge;
pub mod overrides;
pub mod pack;
#[cfg(feature = "audio")]
pub mod prep_corpus;
pub mod train;

#[cfg(feature = "huggingface")]
pub mod push;

pub use import::{run_import, ImportArgs};
pub use merge::{run_merge, MergeArgs};
pub use pack::{run_pack, PackArgs};
#[cfg(feature = "audio")]
pub use prep_corpus::{run_prep_corpus, PrepCorpusArgs};
pub use train::{run_train, TrainArgs};

#[cfg(feature = "huggingface")]
pub use push::{run_push, PushArgs};
