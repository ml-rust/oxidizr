//! Oxidizr: Rust-based ML training framework for GPT-style language models.
//!
//! `main.rs` is a thin CLI entry point over this library. Exposing the
//! modules here (rather than only as `mod` items on the binary) lets
//! `tests/` exercise the real model-construction path — the same
//! `VarBuilder`/`LitGPT::new` wiring `commands::train` uses — through the
//! public API instead of re-implementing it.

pub mod commands;
pub mod config;
pub mod config_convert;
pub mod data;
pub mod distributed;
pub mod model;
pub mod prelude;
pub mod schedule;
pub mod speech_corpus;
pub mod trainer;
pub mod vocab;
