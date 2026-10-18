//! Thin re-exports from boostr 0.3.
//!
//! This module is only a local import surface for oxidizr migration units. It
//! intentionally does not define compatibility structs, wrapper functions, or
//! local implementations of boostr behavior.

pub use boostr::{DType, Init, Tensor, VarBuilder, VarMap};

pub use boostr::error::{Error, Result};
pub use boostr::nn::loss::cross_entropy_loss as cross_entropy;
pub use boostr::nn::loss::cross_entropy_loss_masked as cross_entropy_masked;
pub use boostr::nn::{Embedding, Linear, RmsNorm};
pub use boostr::optimizer::clip_grad_norm as compute_grad_norm;
pub use boostr::optimizer::AdamW as FusedAdamW;
