//! The loss-spike guard: an EMA of the step loss and the rule that flags a spike.
//!
//! The guard reads the step loss after the data-parallel sum, so every rank
//! sees the same value and makes the same decision.

use super::config::{SpikeAction, SpikeGuardConfig};
use crate::config::CheckpointMeta;
use crate::prelude::{Error, Result};
use serde::{Deserialize, Serialize};

/// The EMA and counters of a guard, as a checkpoint meta records them.
///
/// The meta stores each f64 at full precision, so a resume continues bit-identical.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct SpikeGuardState {
    /// EMA of the step loss.
    pub mean: f64,
    /// EMA variance of the step loss.
    pub variance: f64,
    /// Losses the EMA holds. A flagged or non-finite loss never enters it.
    pub observed: u64,
    /// Steps the guard flagged.
    pub detected: u64,
    /// Flagged steps that skipped the optimizer.
    pub skipped: u64,
}

/// One flagged step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpikeEvent {
    /// 1-based step number.
    pub step: usize,
    pub loss: f64,
    /// EMA mean before this step.
    pub mean: f64,
    /// EMA standard deviation before this step.
    pub std: f64,
    pub action: SpikeAction,
}

impl SpikeEvent {
    /// Standard deviations above the mean. Infinite when `std` is 0.
    pub fn z(&self) -> f64 {
        (self.loss - self.mean) / self.std
    }

    /// True when the step skipped the optimizer.
    pub fn skipped(&self) -> bool {
        self.action == SpikeAction::Skip
    }

    /// The stderr line of this event.
    pub fn describe(&self) -> String {
        let outcome = if self.skipped() {
            "The step skips the optimizer and keeps its batches"
        } else {
            "The step trains as usual"
        };
        format!(
            "[SpikeGuard] Step {}: loss {:.4} is {:.1} std above the loss EMA {:.4} (std {:.4}). \
             {outcome}.",
            self.step,
            self.loss,
            self.z(),
            self.mean,
            self.std
        )
    }
}

/// The guard of one run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpikeGuard {
    config: SpikeGuardConfig,
    state: SpikeGuardState,
}

impl SpikeGuard {
    /// A guard with an empty EMA.
    ///
    /// # Errors
    /// Returns an error when a field of `config` is out of range.
    pub fn new(config: SpikeGuardConfig) -> Result<Self> {
        config.check()?;
        Ok(Self {
            config,
            state: SpikeGuardState::default(),
        })
    }

    pub fn config(&self) -> &SpikeGuardConfig {
        &self.config
    }

    pub fn state(&self) -> SpikeGuardState {
        self.state
    }

    /// Continue from a checkpoint's `state`.
    ///
    /// # Errors
    /// Returns an error when the mean or variance is not finite, or the variance is negative.
    pub fn restore(&mut self, state: SpikeGuardState) -> Result<()> {
        if !(state.mean.is_finite() && state.variance.is_finite() && state.variance >= 0.0) {
            return Err(Error::TrainingError {
                reason: format!(
                    "the checkpoint's spike guard state holds mean {} and variance {}. Both \
                     must be finite and the variance 0 or more. Repair the meta's \
                     `spike_guard` key, or remove it to restart the guard.",
                    state.mean, state.variance
                ),
            });
        }
        self.state = state;
        Ok(())
    }

    /// Judges the loss of the 1-based step `step`.
    ///
    /// Returns the event of a flagged step. A finite loss that is not a spike
    /// enters the EMA. A flagged loss and a non-finite loss never do.
    /// The guard flags nothing until the EMA holds
    /// [`warmup_losses`](SpikeGuardConfig::warmup_losses) losses.
    pub fn observe(&mut self, step: usize, loss: f64) -> Option<SpikeEvent> {
        if !loss.is_finite() {
            return None;
        }
        let state = &mut self.state;
        if state.observed >= self.config.warmup_losses() {
            let std = state.variance.sqrt();
            if loss > state.mean + self.config.z_threshold * std {
                state.detected += 1;
                if self.config.action == SpikeAction::Skip {
                    state.skipped += 1;
                }
                return Some(SpikeEvent {
                    step,
                    loss,
                    mean: state.mean,
                    std,
                    action: self.config.action,
                });
            }
        }
        self.update(loss);
        None
    }

    /// Folds `loss` into the EMA mean and variance.
    ///
    /// The first loss sets the mean with variance 0. Each later loss runs the
    /// incremental exponentially weighted update:
    /// `d = x - mean`, `mean += a * d`, `var = (1 - a) * (var + a * d * d)`.
    fn update(&mut self, loss: f64) {
        let state = &mut self.state;
        if state.observed == 0 {
            state.mean = loss;
            state.variance = 0.0;
        } else {
            let alpha = self.config.alpha();
            let diff = loss - state.mean;
            let incr = alpha * diff;
            state.mean += incr;
            state.variance = (1.0 - alpha) * (state.variance + diff * incr);
        }
        state.observed += 1;
    }
}

/// The guard state a run at `start_step` resumes from the meta at `meta_path`.
///
/// A run at step 0, a resume without a meta, and a meta without the key all give `None`.
///
/// # Errors
/// Returns an error when the meta is unreadable.
pub fn read_spike_guard_state(
    meta_path: Option<&str>,
    start_step: usize,
) -> Result<Option<SpikeGuardState>> {
    let Some(path) = meta_path.filter(|_| start_step > 0) else {
        return Ok(None);
    };
    let meta = CheckpointMeta::load(path).map_err(|e| Error::TrainingError {
        reason: format!(
            "checkpoint meta '{path}' was not read: {e}. Restore the file to resume with its \
             spike guard state."
        ),
    })?;
    Ok(meta.spike_guard)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guard(window: usize, z: f64, min_steps: usize, action: SpikeAction) -> SpikeGuard {
        SpikeGuard::new(SpikeGuardConfig {
            window,
            z_threshold: z,
            min_steps,
            action,
        })
        .expect("guard")
    }

    #[test]
    fn the_ema_matches_a_hand_computed_update() {
        let mut g = guard(4, 6.0, 50, SpikeAction::Skip);
        for (step, loss) in [1.0, 2.0, 0.5].into_iter().enumerate() {
            assert!(g.observe(step + 1, loss).is_none());
        }
        // a = 0.4. After 1.0: mean 1, var 0.
        // After 2.0: d = 1, mean 1.4, var 0.6 * (0 + 0.4) = 0.24.
        // After 0.5: d = -0.9, mean 1.04, var 0.6 * (0.24 + 0.324) = 0.3384.
        let s = g.state();
        assert_eq!(s.observed, 3);
        assert!((s.mean - 1.04).abs() < 1e-12, "{}", s.mean);
        assert!((s.variance - 0.3384).abs() < 1e-12, "{}", s.variance);
    }

    #[test]
    fn nothing_is_flagged_before_min_steps() {
        let mut g = guard(4, 1.0, 3, SpikeAction::Skip);
        assert!(g.observe(1, 1.0).is_none());
        assert!(g.observe(2, 1.1).is_none());
        assert!(
            g.observe(3, 50.0).is_none(),
            "two losses are below min_steps 3"
        );
        assert_eq!(g.state().observed, 3);
    }

    #[test]
    fn a_spike_skips_counts_and_stays_out_of_the_ema() {
        let mut g = guard(4, 6.0, 2, SpikeAction::Skip);
        for (step, loss) in [1.0, 1.1, 1.0, 1.1].into_iter().enumerate() {
            assert!(g.observe(step + 1, loss).is_none(), "step {}", step + 1);
        }
        let before = g.state();
        let event = g.observe(5, 1000.0).expect("a spike");
        assert!(event.skipped());
        assert_eq!(event.step, 5);
        assert_eq!(event.mean, before.mean);
        assert!(event.z() > 6.0);
        let after = g.state();
        assert_eq!((after.mean, after.variance), (before.mean, before.variance));
        assert_eq!((after.observed, after.detected, after.skipped), (4, 1, 1));
    }

    #[test]
    fn warn_counts_a_detection_and_no_skip() {
        let mut g = guard(4, 6.0, 2, SpikeAction::Warn);
        g.observe(1, 1.0);
        g.observe(2, 1.1);
        let event = g.observe(3, 1000.0).expect("a spike");
        assert!(!event.skipped());
        assert_eq!((g.state().detected, g.state().skipped), (1, 0));
    }

    #[test]
    fn a_non_finite_loss_is_never_judged_and_never_enters_the_ema() {
        let mut g = guard(4, 6.0, 2, SpikeAction::Skip);
        g.observe(1, 1.0);
        g.observe(2, 1.1);
        let before = g.state();
        for loss in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(g.observe(3, loss).is_none());
        }
        assert_eq!(g.state(), before);
    }

    #[test]
    fn restore_continues_the_ema_and_rejects_a_bad_state() {
        let mut a = guard(4, 6.0, 2, SpikeAction::Skip);
        for (step, loss) in [1.0, 1.2, 0.9].into_iter().enumerate() {
            a.observe(step + 1, loss);
        }
        let mut b = guard(4, 6.0, 2, SpikeAction::Skip);
        b.restore(a.state()).expect("restore");
        a.observe(4, 1.05);
        b.observe(4, 1.05);
        assert_eq!(a.state(), b.state());
        let bad = SpikeGuardState {
            variance: -1.0,
            ..SpikeGuardState::default()
        };
        assert!(b.restore(bad).is_err());
    }

    #[test]
    fn the_state_round_trips_through_json_bit_for_bit() {
        let state = SpikeGuardState {
            mean: 0.1 + 0.2,
            variance: 1.0 / 3.0,
            observed: 7,
            detected: 2,
            skipped: 1,
        };
        let json = serde_json::to_string(&state).expect("serialize");
        let back: SpikeGuardState = serde_json::from_str(&json).expect("parse");
        assert_eq!(back.mean.to_bits(), state.mean.to_bits());
        assert_eq!(back.variance.to_bits(), state.variance.to_bits());
        assert_eq!(back, state);
    }
}
