//! Learning-rate decay schedules applied after warmup.
//!
//! `LitTrainer` runs linear warmup from step 0 to `warmup_steps` unconditionally
//! (see `trainer.rs`). This module owns what happens AFTER warmup: hold the
//! base learning rate flat (`Constant`, the default — byte-identical to the
//! prior behavior), or decay it toward `min_lr_ratio * base_lr` by the run's
//! last step (`Cosine` / `Linear`).

use serde::{Deserialize, Serialize};

/// Post-warmup learning-rate schedule. `Constant` is the default so an
/// existing config with no `lr_schedule` key keeps its current behavior.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LrSchedule {
    /// Hold the base learning rate flat after warmup.
    #[default]
    Constant,
    /// Cosine decay from the base learning rate to `min_lr_ratio * base_lr`.
    Cosine,
    /// Linear decay from the base learning rate to `min_lr_ratio * base_lr`.
    Linear,
}

impl LrSchedule {
    /// Whether this schedule decays toward an end step, and therefore needs
    /// a finite total step count to decay across. `Constant` never does.
    pub fn needs_total_steps(&self) -> bool {
        !matches!(self, LrSchedule::Constant)
    }

    /// Multiplier applied to the base learning rate at `step` (0-indexed).
    ///
    /// `warmup_steps` is the step warmup ends at (the caller drives warmup
    /// itself; this only decides what happens from that point on).
    /// `total_steps` is the run's last step. `min_lr_ratio` is the fraction
    /// of the base learning rate the schedule decays to by `total_steps`.
    ///
    /// Returns `1.0` for every step under `Constant`. For `Cosine`/`Linear`,
    /// returns `1.0` at and before `warmup_steps`, `min_lr_ratio` at and past
    /// `total_steps`, and interpolates between.
    pub fn decay_multiplier(
        &self,
        step: usize,
        warmup_steps: usize,
        total_steps: usize,
        min_lr_ratio: f64,
    ) -> f64 {
        if *self == LrSchedule::Constant {
            return 1.0;
        }
        if step <= warmup_steps {
            return 1.0;
        }
        // No decay span to work with: hold at the floor rather than divide
        // by zero. `LitConfig::validate` rejects a schedule with no total
        // steps before this is ever reached in a real run.
        if total_steps <= warmup_steps {
            return min_lr_ratio;
        }

        let progress =
            ((step - warmup_steps) as f64 / (total_steps - warmup_steps) as f64).min(1.0);
        let shape = match self {
            LrSchedule::Cosine => 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()),
            LrSchedule::Linear => 1.0 - progress,
            LrSchedule::Constant => 1.0,
        };
        min_lr_ratio + (1.0 - min_lr_ratio) * shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_never_decays() {
        for step in [0, 10, 50, 100, 1000] {
            assert_eq!(
                LrSchedule::Constant.decay_multiplier(step, 10, 100, 0.1),
                1.0
            );
        }
    }

    #[test]
    fn constant_does_not_need_total_steps() {
        assert!(!LrSchedule::Constant.needs_total_steps());
        assert!(LrSchedule::Cosine.needs_total_steps());
        assert!(LrSchedule::Linear.needs_total_steps());
    }

    #[test]
    fn cosine_at_warmup_end_is_full_lr() {
        let m = LrSchedule::Cosine.decay_multiplier(10, 10, 110, 0.1);
        assert_eq!(m, 1.0);
    }

    #[test]
    fn cosine_at_midpoint_is_halfway_between_one_and_min_ratio() {
        // warmup=0, total=100, step=50 -> progress=0.5 -> cos(pi/2)=0 -> shape=0.5
        let min_ratio = 0.1;
        let m = LrSchedule::Cosine.decay_multiplier(50, 0, 100, min_ratio);
        let expected = min_ratio + (1.0 - min_ratio) * 0.5;
        assert!((m - expected).abs() < 1e-9, "got {m}, expected {expected}");
    }

    #[test]
    fn cosine_at_total_steps_is_min_lr_ratio() {
        let m = LrSchedule::Cosine.decay_multiplier(100, 0, 100, 0.1);
        assert!((m - 0.1).abs() < 1e-9, "got {m}");
    }

    #[test]
    fn cosine_past_total_steps_stays_at_min_lr_ratio() {
        let m = LrSchedule::Cosine.decay_multiplier(500, 0, 100, 0.1);
        assert!((m - 0.1).abs() < 1e-9, "got {m}");
    }

    #[test]
    fn linear_at_warmup_end_is_full_lr() {
        let m = LrSchedule::Linear.decay_multiplier(10, 10, 110, 0.0);
        assert_eq!(m, 1.0);
    }

    #[test]
    fn linear_at_midpoint_is_the_average_of_one_and_min_ratio() {
        let min_ratio = 0.2;
        let m = LrSchedule::Linear.decay_multiplier(50, 0, 100, min_ratio);
        let expected = min_ratio + (1.0 - min_ratio) * 0.5;
        assert!((m - expected).abs() < 1e-9, "got {m}, expected {expected}");
    }

    #[test]
    fn linear_at_total_steps_is_min_lr_ratio() {
        let m = LrSchedule::Linear.decay_multiplier(100, 0, 100, 0.2);
        assert!((m - 0.2).abs() < 1e-9, "got {m}");
    }

    #[test]
    fn schedule_with_warmup_decays_over_the_post_warmup_span_only() {
        // warmup=10, total=110 -> post-warmup span is 100 steps.
        // step=60 is 50 steps past warmup -> progress=0.5.
        let min_ratio = 0.0;
        let m = LrSchedule::Linear.decay_multiplier(60, 10, 110, min_ratio);
        assert!((m - 0.5).abs() < 1e-9, "got {m}");
    }

    #[test]
    fn a_schedule_with_no_decay_span_holds_at_the_floor() {
        // total_steps == warmup_steps: nothing to decay across.
        let m = LrSchedule::Cosine.decay_multiplier(150, 100, 100, 0.1);
        assert!((m - 0.1).abs() < 1e-9, "got {m}");
    }
}
