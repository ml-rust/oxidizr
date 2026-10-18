//! The `trainer.loss_scale` YAML block and the rule that decides when it applies.

use crate::config::DTypePrecision;
use crate::prelude::{Error, Result};
use boostr::optimizer::GradScalerConfig;
use serde::{Deserialize, Serialize};

fn default_initial() -> f64 {
    GradScalerConfig::default().initial_scale
}
fn default_growth_factor() -> f64 {
    GradScalerConfig::default().growth_factor
}
fn default_backoff_factor() -> f64 {
    GradScalerConfig::default().backoff_factor
}
fn default_growth_interval() -> u64 {
    GradScalerConfig::default().growth_interval
}

/// Dynamic loss scaling for an f16 run. Every field is optional.
///
/// The trainer multiplies the loss by the scale before backward. It divides
/// every gradient by the scale before clipping. A non-finite gradient norm
/// skips the step and multiplies the scale by `backoff_factor`.
/// `growth_interval` clean steps in a row multiply it by `growth_factor`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LossScaleConfig {
    /// Starting scale. Default 65536.
    #[serde(default = "default_initial")]
    pub initial: f64,
    /// Scale multiplier after `growth_interval` clean steps. Default 2.
    #[serde(default = "default_growth_factor")]
    pub growth_factor: f64,
    /// Scale multiplier on an overflow step. Default 0.5.
    #[serde(default = "default_backoff_factor")]
    pub backoff_factor: f64,
    /// Clean steps in a row before the scale grows. Default 2000.
    #[serde(default = "default_growth_interval")]
    pub growth_interval: u64,
}

impl Default for LossScaleConfig {
    fn default() -> Self {
        Self {
            initial: default_initial(),
            growth_factor: default_growth_factor(),
            backoff_factor: default_backoff_factor(),
            growth_interval: default_growth_interval(),
        }
    }
}

impl LossScaleConfig {
    /// The boostr scaler config of this block.
    pub fn scaler_config(&self) -> GradScalerConfig {
        GradScalerConfig {
            initial_scale: self.initial,
            growth_factor: self.growth_factor,
            backoff_factor: self.backoff_factor,
            growth_interval: self.growth_interval,
        }
    }

    /// Check every field against its range, naming the YAML key on an error.
    ///
    /// # Errors
    /// Returns an error naming the first key out of range.
    pub fn check(&self) -> Result<()> {
        let bad = |key: &str, value: String, rule: &str| {
            Err(Error::TrainingError {
                reason: format!(
                    "trainer.loss_scale.{key} is {value}: it must be {rule}. Fix the value, or \
                     remove the key to use its default."
                ),
            })
        };
        if !(self.initial.is_finite() && self.initial > 0.0) {
            return bad("initial", self.initial.to_string(), "finite and positive");
        }
        if !(self.growth_factor.is_finite() && self.growth_factor > 1.0) {
            return bad(
                "growth_factor",
                self.growth_factor.to_string(),
                "finite and above 1",
            );
        }
        if !(self.backoff_factor > 0.0 && self.backoff_factor < 1.0) {
            return bad(
                "backoff_factor",
                self.backoff_factor.to_string(),
                "above 0 and below 1",
            );
        }
        if self.growth_interval == 0 {
            return bad("growth_interval", "0".to_string(), "at least 1");
        }
        Ok(())
    }
}

/// The loss scaling a run with model precision `dtype` uses.
///
/// - f16 uses `configured`, or the defaults when the block is absent.
/// - bf16 and f32 use none. A configured block for them is an error: both
///   share f32's exponent range, so their gradients do not underflow.
///
/// # Errors
/// Returns an error for a block under bf16 or f32, or a field out of range.
pub fn resolve_loss_scale(
    configured: Option<&LossScaleConfig>,
    dtype: DTypePrecision,
) -> Result<Option<LossScaleConfig>> {
    match (dtype, configured) {
        (DTypePrecision::F16, configured) => {
            let config = configured.copied().unwrap_or_default();
            config.check()?;
            Ok(Some(config))
        }
        (_, None) => Ok(None),
        (other, Some(_)) => Err(Error::TrainingError {
            reason: format!(
                "trainer.loss_scale is set, but model.dtype is {other}. Loss scaling applies \
                 to f16 only. Remove trainer.loss_scale, or set model.dtype: f16."
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_without_a_block_uses_the_defaults() {
        let resolved = resolve_loss_scale(None, DTypePrecision::F16).expect("resolve");
        let config = resolved.expect("f16 scales by default");
        assert_eq!(config, LossScaleConfig::default());
        assert_eq!(config.initial, 65536.0);
        assert_eq!(config.growth_factor, 2.0);
        assert_eq!(config.backoff_factor, 0.5);
        assert_eq!(config.growth_interval, 2000);
    }

    #[test]
    fn f16_keeps_a_configured_block() {
        let configured = LossScaleConfig {
            initial: 1024.0,
            ..LossScaleConfig::default()
        };
        let resolved = resolve_loss_scale(Some(&configured), DTypePrecision::F16).expect("ok");
        assert_eq!(resolved, Some(configured));
    }

    #[test]
    fn bf16_and_f32_without_a_block_use_none() {
        for dtype in [DTypePrecision::BF16, DTypePrecision::F32] {
            assert_eq!(resolve_loss_scale(None, dtype).expect("ok"), None);
        }
    }

    #[test]
    fn a_block_under_bf16_or_f32_is_an_error_naming_key_and_dtype() {
        for (dtype, name) in [(DTypePrecision::BF16, "bf16"), (DTypePrecision::F32, "f32")] {
            let err = resolve_loss_scale(Some(&LossScaleConfig::default()), dtype)
                .expect_err("loss_scale must be refused");
            let text = err.to_string();
            assert!(text.contains("trainer.loss_scale"), "{text}");
            assert!(text.contains(name), "{text}");
        }
    }

    #[test]
    fn partial_yaml_fills_the_other_fields() {
        let config: LossScaleConfig = serde_yaml::from_str("growth_interval: 10\n").expect("parse");
        assert_eq!(
            config,
            LossScaleConfig {
                growth_interval: 10,
                ..LossScaleConfig::default()
            }
        );
        let empty: LossScaleConfig = serde_yaml::from_str("{}").expect("parse");
        assert_eq!(empty, LossScaleConfig::default());
    }

    #[test]
    fn out_of_range_fields_name_their_key() {
        let cases = [
            (
                "initial",
                LossScaleConfig {
                    initial: 0.0,
                    ..Default::default()
                },
            ),
            (
                "initial",
                LossScaleConfig {
                    initial: f64::INFINITY,
                    ..Default::default()
                },
            ),
            (
                "growth_factor",
                LossScaleConfig {
                    growth_factor: 1.0,
                    ..Default::default()
                },
            ),
            (
                "backoff_factor",
                LossScaleConfig {
                    backoff_factor: 1.0,
                    ..Default::default()
                },
            ),
            (
                "growth_interval",
                LossScaleConfig {
                    growth_interval: 0,
                    ..Default::default()
                },
            ),
        ];
        for (key, config) in cases {
            let err =
                resolve_loss_scale(Some(&config), DTypePrecision::F16).expect_err("out of range");
            assert!(
                err.to_string()
                    .contains(&format!("trainer.loss_scale.{key}")),
                "{err}"
            );
        }
    }
}
