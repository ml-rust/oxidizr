//! Prelude module for consistent imports throughout oxidizr
//!
//! This module re-exports types from boostr with Candle-compatible names.
//! Since oxidizr is CUDA-focused, we provide concrete CUDA types.

// CudaRuntime from boostr's cubecl_cuda re-export
#[cfg(feature = "cuda")]
pub use boostr::cubecl_cuda::CudaRuntime;

// Core tensor types - use CUDA runtime by default
pub use boostr::{D, DType, Device};

// Error alias for Candle compatibility
pub use boostr::TensorError as Error;

// Re-export Result (boostr's Result type)
pub use boostr::Result;

// Default Tensor type for oxidizr (CUDA when enabled)
// This makes migration from Candle easier - code can use `Tensor` directly
#[cfg(feature = "cuda")]
pub type Tensor = boostr::Tensor<CudaRuntime>;

// Non-CUDA builds will fail to compile (oxidizr requires CUDA)
#[cfg(not(feature = "cuda"))]
compile_error!("oxidizr requires the 'cuda' feature to be enabled");

// Neural network layers (generic)
pub use boostr::Conv1dConfig;

// CUDA-specific VarBuilder and VarMap (for convenience)
#[cfg(feature = "cuda")]
pub type VarBuilder = boostr::VarBuilder<CudaRuntime>;

#[cfg(feature = "cuda")]
pub type VarMap = boostr::VarMap<CudaRuntime>;

// Layer types (CUDA-specific type aliases)
#[cfg(feature = "cuda")]
pub type Linear = boostr::Linear<CudaRuntime>;

#[cfg(feature = "cuda")]
pub type Conv1d = boostr::Conv1d<CudaRuntime>;

#[cfg(feature = "cuda")]
pub type Embedding = boostr::Embedding<CudaRuntime>;

#[cfg(feature = "cuda")]
pub type LayerNorm = boostr::LayerNorm<CudaRuntime>;

#[cfg(feature = "cuda")]
pub type RmsNorm = boostr::RmsNorm<CudaRuntime>;

// Convenience factory functions (need CudaRuntime specified)
pub use boostr::{conv1d, embedding, layer_norm, linear, linear_no_bias, rms_norm};

// Loss functions
pub use boostr::cross_entropy;

// MoE auxiliary losses (router z-loss prevents softmax saturation)
#[cfg(feature = "cuda")]
pub use boostr::router_z_loss;

// Fused AdamW optimizer (recommended - uses GPU kernels with gradient clamping)
#[cfg(feature = "cuda")]
pub use boostr::FusedAdamW;

// Efficient gradient norm computation (accumulates on GPU, single CPU transfer)
#[cfg(feature = "cuda")]
pub use boostr::compute_grad_norm;

// Global tape for automatic gradient tracking
#[cfg(feature = "cuda")]
pub use boostr::global_tape;

// CUDA-specific GradStore types
#[cfg(feature = "cuda")]
pub type GradStore = boostr::GradStore<Tensor>;

// Candle-compatible ops module
#[cfg(feature = "cuda")]
pub mod ops {
    use super::Tensor;
    use crate::prelude::Result;

    /// Apply SiLU (Sigmoid Linear Unit) activation
    pub fn silu(x: &Tensor) -> Result<Tensor> {
        x.silu()
    }

    /// Apply softmax over the last dimension
    pub fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
        let last_dim = x.rank() as i32 - 1;
        x.softmax(last_dim)
    }

    /// Apply sigmoid activation
    pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
        x.sigmoid()
    }

    /// Fused scaled causal softmax for attention
    ///
    /// Computes softmax(scale * x) with causal masking in one fused operation.
    /// Designed for attention score tensors of shape [batch, heads, seq_q, seq_k].
    ///
    /// Benefits:
    /// - Numerically stable (never creates -inf values)
    /// - Efficient (fuses scale, mask, and softmax in one pass)
    /// - Gradient-stable backward pass
    ///
    /// # Arguments
    /// * `x` - Attention scores [batch, heads, seq_q, seq_k] (Q @ K^T, NOT scaled)
    /// * `scale` - Typically 1/sqrt(head_dim)
    pub fn scaled_causal_softmax(x: &Tensor, scale: f32) -> Result<Tensor> {
        x.scaled_causal_softmax(scale)
    }
}
