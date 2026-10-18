//! Oxidizr: Rust-based ML training framework for GPT-style language models.
//!
//! `main.rs` is a thin CLI entry point over this library. Exposing the
//! modules here (rather than only as `mod` items on the binary) lets
//! `tests/` exercise the real model-construction path — the same
//! `VarBuilder`/`LitGPT::new` wiring `commands::train` uses — through the
//! public API instead of re-implementing it.

pub mod checkpoint_config;
pub mod checkpoint_names;
pub mod checkpoint_set;
pub mod checkpoint_sidecars;
pub mod commands;
pub mod config;
pub mod config_convert;
pub mod config_paths;
pub mod data;
pub mod data_hash;
pub mod distributed;
pub mod dp;
pub mod eval;
pub mod fit;
pub mod grad_clip;
pub mod loss_scale;
pub mod model;
mod oom_guard;
pub mod prelude;
pub mod run_record;
pub mod schedule;
pub mod speech_corpus;
pub mod step_loss;
pub mod trainer;
pub mod vocab;
pub mod weight_decay;
