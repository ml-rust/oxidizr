//! The `trainer.metrics` and `trainer.spike_guard` YAML blocks.

use crate::prelude::{Error, Result};
use serde::{Deserialize, Serialize};

/// Throughput reporting of a run.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricsConfig {
    /// Peak dense throughput of one device in TFLOP/s, at the precision the run trains in.
    ///
    /// Absent, the run reports no MFU. oxidizr never guesses a peak.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_tflops: Option<f64>,
}

impl MetricsConfig {
    /// # Errors
    /// Returns an error when `peak_tflops` is set and not finite and positive.
    pub fn check(&self) -> Result<()> {
        match self.peak_tflops {
            Some(peak) if !(peak.is_finite() && peak > 0.0) => Err(Error::TrainingError {
                reason: format!(
                    "trainer.metrics.peak_tflops is {peak}: it must be a finite positive \
                     number. Set the device's peak TFLOP/s, or remove the key to report no MFU."
                ),
            }),
            _ => Ok(()),
        }
    }
}

/// What the guard does with a step whose loss it flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpikeAction {
    /// Skip the optimizer step. The step still draws its batches.
    #[default]
    Skip,
    /// Report the spike and step as usual.
    Warn,
}

impl SpikeAction {
    /// The YAML name of the action.
    pub fn name(self) -> &'static str {
        match self {
            Self::Skip => "skip",
            Self::Warn => "warn",
        }
    }
}

fn default_window() -> usize {
    100
}
fn default_z_threshold() -> f64 {
    6.0
}
fn default_min_steps() -> usize {
    50
}

/// The loss-spike guard. Every field is optional.
///
/// The guard keeps an exponential moving mean and variance of the step loss.
/// A finite loss above `mean + z_threshold * std` is a spike.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpikeGuardConfig {
    /// EMA span in steps. The EMA weight of a new loss is `2 / (window + 1)`. Default 100.
    #[serde(default = "default_window")]
    pub window: usize,
    /// Standard deviations above the mean that flag a spike. Default 6.0.
    #[serde(default = "default_z_threshold")]
    pub z_threshold: f64,
    /// Losses the EMA must hold before the guard flags any step. Default 50.
    ///
    /// The guard needs at least 2 losses for a variance, so 0 and 1 act as 2.
    #[serde(default = "default_min_steps")]
    pub min_steps: usize,
    /// Default `skip`.
    #[serde(default)]
    pub action: SpikeAction,
}

impl Default for SpikeGuardConfig {
    fn default() -> Self {
        Self {
            window: default_window(),
            z_threshold: default_z_threshold(),
            min_steps: default_min_steps(),
            action: SpikeAction::default(),
        }
    }
}

impl SpikeGuardConfig {
    /// EMA weight of a new loss: `2 / (window + 1)`.
    pub fn alpha(&self) -> f64 {
        2.0 / (self.window as f64 + 1.0)
    }

    /// Losses the EMA must hold before the guard acts. At least 2.
    pub fn warmup_losses(&self) -> u64 {
        self.min_steps.max(2) as u64
    }

    /// # Errors
    /// Returns an error when `window` is 0, or `z_threshold` is not finite and positive.
    pub fn check(&self) -> Result<()> {
        if self.window == 0 {
            return Err(Error::TrainingError {
                reason: "trainer.spike_guard.window is 0: the EMA span must be at least 1 step. \
                         Set a positive window."
                    .to_string(),
            });
        }
        let z = self.z_threshold;
        if !(z.is_finite() && z > 0.0) {
            return Err(Error::TrainingError {
                reason: format!(
                    "trainer.spike_guard.z_threshold is {z}: it must be a finite positive \
                     number. Set a positive threshold, such as 6.0."
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_block_takes_the_documented_defaults() {
        let config: SpikeGuardConfig = serde_yaml::from_str("{}").expect("parse");
        assert_eq!(config, SpikeGuardConfig::default());
        assert_eq!(
            (config.window, config.z_threshold, config.min_steps),
            (100, 6.0, 50)
        );
        assert_eq!(config.action, SpikeAction::Skip);
        let warn: SpikeGuardConfig = serde_yaml::from_str("action: warn").expect("parse");
        assert_eq!(warn.action, SpikeAction::Warn);
    }

    #[test]
    fn an_unknown_key_is_an_error() {
        assert!(serde_yaml::from_str::<SpikeGuardConfig>("windw: 3").is_err());
        assert!(serde_yaml::from_str::<MetricsConfig>("peak: 3").is_err());
    }

    #[test]
    fn bad_values_name_their_key() {
        let zero = SpikeGuardConfig {
            window: 0,
            ..SpikeGuardConfig::default()
        };
        assert!(zero
            .check()
            .is_err_and(|e| e.to_string().contains("window")));
        for z in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let bad = SpikeGuardConfig {
                z_threshold: z,
                ..SpikeGuardConfig::default()
            };
            assert!(bad
                .check()
                .is_err_and(|e| e.to_string().contains("z_threshold")));
        }
        for peak in [0.0, -2.0, f64::NAN] {
            let bad = MetricsConfig {
                peak_tflops: Some(peak),
            };
            assert!(bad
                .check()
                .is_err_and(|e| e.to_string().contains("peak_tflops")));
        }
        assert!(MetricsConfig::default().check().is_ok());
    }

    #[test]
    fn the_ema_weight_follows_the_span() {
        let config = SpikeGuardConfig {
            window: 4,
            ..SpikeGuardConfig::default()
        };
        assert_eq!(config.alpha(), 0.4);
        let early = SpikeGuardConfig {
            min_steps: 0,
            ..SpikeGuardConfig::default()
        };
        assert_eq!(early.warmup_losses(), 2);
    }
}
