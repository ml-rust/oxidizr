//! Thin re-exports from boostr 0.3.
//!
//! This module is only a local import surface for oxidizr migration units. It
//! intentionally does not define compatibility structs, wrapper functions, or
//! local implementations of boostr behavior.

pub use boostr::{DType, Init, Tensor, VarBuilder, VarMap, Weight};

#[cfg(feature = "cuda")]
pub use boostr::CudaRuntime;

pub use boostr::error::{Error, Result};
pub use boostr::nn::loss::cross_entropy_loss as cross_entropy;
pub use boostr::nn::{Conv1d, Embedding, LayerNorm, Linear, RmsNorm};
pub use boostr::optimizer::clip_grad_norm as compute_grad_norm;
pub use boostr::optimizer::{AdamW as FusedAdamW, AdamWConfig};
pub use boostr::router_z_loss;
pub use boostr::runtime::Device;
