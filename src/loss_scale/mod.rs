//! Dynamic loss scaling for f16 training.
//!
//! f16 has a 5-bit exponent, so small gradients flush to zero. An f16 run
//! scales its loss up before backward and its gradients down before
//! clipping. bf16 and f32 share f32's exponent range and need no scaling.

mod config;
mod resume;
mod scaler;

pub use config::{resolve_loss_scale, LossScaleConfig};
pub use resume::read_loss_scale_state;
pub use scaler::{restore_loss_scale, weighted_loss, LossScaler};
