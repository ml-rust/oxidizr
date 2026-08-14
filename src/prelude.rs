//! Thin re-exports from boostr 0.3.
//!
//! This module is only a local import surface for oxidizr migration units. It
//! intentionally does not define compatibility structs, wrapper functions, or
//! local implementations of boostr behavior.

pub use boostr::{DType, Init, Tensor, VarBuilder, VarMap};

#[cfg(feature = "cuda")]
pub use boostr::CudaRuntime;

pub use boostr::error::{Error, Result};
pub use boostr::nn::loss::cross_entropy_loss as cross_entropy;
pub use boostr::nn::{Embedding, Linear, RmsNorm};
pub use boostr::optimizer::clip_grad_norm as compute_grad_norm;
pub use boostr::optimizer::AdamW as FusedAdamW;
// Only the NCCL synchronizer names this trait, and that module is feature-gated;
// re-exporting it unconditionally leaves it unused in a default build.
#[cfg(feature = "nccl")]
pub use boostr::runtime::Device;
