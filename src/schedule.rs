//! Learning rate of every training step.
//!
//! [`LrSchedule`] is the YAML `trainer.lr_schedule` key. [`LrPlan`] maps it
//! onto `boostr::optimizer::LrSchedule` and answers the rate of one step.
//! Linear warmup covers steps `0..warmup_steps`. The post-warmup schedule
//! covers every later step, as a multiplier of `trainer.learning_rate`.

use crate::config::TrainerConfig;
use crate::prelude::{Error, Result};
use boostr::optimizer::{DecayShape, LrSchedule as BoostrSchedule};
use serde::{Deserialize, Serialize};

/// Fraction of the run's steps a `wsd` schedule decays across when neither
/// `trainer.decay_ratio` nor `trainer.decay_steps` is set.
pub const DEFAULT_DECAY_RATIO: f64 = 0.1;

/// Smallest learning rate a warmup step applies.
const MIN_WARMUP_LR: f64 = 1e-7;

/// Post-warmup learning-rate schedule. `Constant` is the default, so a config
/// with no `lr_schedule` key holds the base learning rate after warmup.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LrSchedule {
    /// Hold the base learning rate flat after warmup.
    #[default]
    Constant,
    /// Cosine decay from the end of warmup to `min_lr_ratio * base_lr` at the last step.
    Cosine,
    /// Linear decay from the end of warmup to `min_lr_ratio * base_lr` at the last step.
    Linear,
    /// Warmup-stable-decay: hold the base rate, then decay over the last
    /// `decay_steps` (or `decay_ratio` of all) steps to `min_lr_ratio * base_lr`.
    Wsd,
}

/// Curve of the `wsd` decay phase (`trainer.decay_shape`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LrDecayShape {
    /// `1 - progress`.
    #[default]
    Linear,
    /// `0.5 * (1 + cos(pi * progress))`.
    Cosine,
    /// `1 - sqrt(progress)`.
    Sqrt,
}

impl From<LrDecayShape> for DecayShape {
    fn from(shape: LrDecayShape) -> Self {
        match shape {
            LrDecayShape::Linear => DecayShape::Linear,
            LrDecayShape::Cosine => DecayShape::Cosine,
            LrDecayShape::Sqrt => DecayShape::Sqrt,
        }
    }
}

impl LrSchedule {
    /// Whether this schedule decays toward an end step. `Constant` never does.
    pub fn needs_total_steps(&self) -> bool {
        !matches!(self, LrSchedule::Constant)
    }

    /// The post-warmup multiplier of the base learning rate as a boostr schedule.
    ///
    /// `get_lr(step)` of the result is `1.0` up to the decay start and
    /// `config.min_lr_ratio` at and past `total_steps`. `Cosine` and `Linear`
    /// decay across `warmup_steps..total_steps`. A run with `total_steps <=
    /// warmup_steps` has no decay span: every step past `warmup_steps` sits at
    /// the floor.
    ///
    /// Returns an error when a `wsd` decay span does not fit after warmup.
    pub fn to_boostr(
        self,
        config: &TrainerConfig,
        warmup_steps: usize,
        total_steps: usize,
    ) -> Result<BoostrSchedule> {
        let wsd =
            |stable_steps: usize, decay_steps: usize, shape: DecayShape| BoostrSchedule::Wsd {
                base_lr: 1.0,
                min_lr: config.min_lr_ratio,
                warmup_steps: 0,
                stable_steps: stable_steps as u64,
                decay_steps: decay_steps as u64,
                decay_shape: shape,
            };
        let after_warmup = |shape: DecayShape| {
            if total_steps > warmup_steps {
                wsd(warmup_steps, total_steps - warmup_steps, shape)
            } else {
                wsd(warmup_steps.saturating_add(1), 0, shape)
            }
        };
        match self {
            LrSchedule::Constant => Ok(BoostrSchedule::Constant { lr: 1.0 }),
            LrSchedule::Cosine => Ok(after_warmup(DecayShape::Cosine)),
            LrSchedule::Linear => Ok(after_warmup(DecayShape::Linear)),
            LrSchedule::Wsd => {
                let decay_steps = wsd_decay_steps(config, total_steps);
                if warmup_steps.saturating_add(decay_steps) > total_steps {
                    return Err(Error::InvalidArgument {
                        arg: "trainer.decay_steps",
                        reason: format!(
                            "lr_schedule: wsd needs {warmup_steps} warmup steps plus \
                             {decay_steps} decay steps, but the run has {total_steps} steps. \
                             Lower trainer.decay_steps, trainer.decay_ratio or the warmup, \
                             or raise trainer.max_steps."
                        ),
                    });
                }
                let shape = config.decay_shape.unwrap_or_default().into();
                Ok(wsd(total_steps - decay_steps, decay_steps, shape))
            }
        }
    }
}

/// Decay span of a `wsd` schedule over a run of `total_steps` steps.
fn wsd_decay_steps(config: &TrainerConfig, total_steps: usize) -> usize {
    match config.decay_steps {
        Some(steps) => steps,
        None => {
            let ratio = config.decay_ratio.unwrap_or(DEFAULT_DECAY_RATIO);
            (total_steps as f64 * ratio).ceil() as usize
        }
    }
}

/// Learning rate of every step of one run.
pub struct LrPlan {
    base_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    warmup: BoostrSchedule,
    after_warmup: BoostrSchedule,
}

impl LrPlan {
    /// The plan of a run whose epochs hold `epoch_steps` optimizer steps.
    ///
    /// Warmup is `trainer.warmup_steps`, else `ceil(warmup_ratio * epoch_steps)`,
    /// else 0. The run's last step is `epoch_steps`, capped by a nonzero
    /// `trainer.max_steps`.
    pub fn new(config: &TrainerConfig, epoch_steps: usize) -> Result<Self> {
        let warmup_steps = match (config.warmup_steps, config.warmup_ratio) {
            (Some(steps), _) => steps,
            (None, Some(ratio)) => (epoch_steps as f64 * ratio).ceil() as usize,
            (None, None) => 0,
        };
        let total_steps = if config.max_steps == 0 {
            epoch_steps
        } else {
            config.max_steps.min(epoch_steps)
        };
        let after_warmup = config
            .lr_schedule
            .to_boostr(config, warmup_steps, total_steps)?;
        Ok(Self {
            base_lr: config.learning_rate,
            warmup_steps,
            total_steps,
            warmup: BoostrSchedule::LinearWarmup {
                base_lr: config.learning_rate,
                warmup_steps: warmup_steps as u64,
            },
            after_warmup,
        })
    }

    /// Learning rate the optimizer applies in 0-indexed step `step`.
    ///
    /// Warmup step `s` applies `base_lr * (s + 1) / warmup_steps`, at least
    /// `1e-7`. A later step applies `base_lr` times the post-warmup multiplier.
    pub fn lr_at(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            self.warmup.get_lr(step as u64 + 1).max(MIN_WARMUP_LR)
        } else {
            self.base_lr * self.after_warmup.get_lr(step as u64)
        }
    }

    /// Steps of linear warmup. 0 disables warmup.
    pub fn warmup_steps(&self) -> usize {
        self.warmup_steps
    }

    /// Learning rate warmup climbs to, and the post-warmup multipliers scale.
    pub fn base_lr(&self) -> f64 {
        self.base_lr
    }

    /// The run's step count: the loop runs steps `0..total_steps`.
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }
}

/// Checks the schedule keys of `trainer` that need no data to check.
pub fn validate(trainer: &TrainerConfig) -> anyhow::Result<()> {
    let schedule = trainer.lr_schedule;
    if schedule.needs_total_steps() && trainer.max_steps == 0 && trainer.num_epochs == 0 {
        anyhow::bail!(
            "trainer.lr_schedule = {schedule:?} decays the learning rate toward a total step \
             count, but neither trainer.max_steps nor trainer.num_epochs names one \
             (both are 0). Set trainer.max_steps to the run's last step, or \
             trainer.num_epochs to a positive number of epochs."
        );
    }
    if !(0.0..=1.0).contains(&trainer.min_lr_ratio) {
        anyhow::bail!(
            "trainer.min_lr_ratio ({}) must be between 0.0 and 1.0: it names the \
             fraction of learning_rate the schedule decays to.",
            trainer.min_lr_ratio
        );
    }
    let has_wsd_keys = trainer.decay_ratio.is_some()
        || trainer.decay_steps.is_some()
        || trainer.decay_shape.is_some();
    if schedule != LrSchedule::Wsd {
        if has_wsd_keys {
            anyhow::bail!(
                "trainer.decay_ratio, trainer.decay_steps and trainer.decay_shape apply only to \
                 lr_schedule: wsd, but lr_schedule is {schedule:?}. Remove them, or set \
                 trainer.lr_schedule: wsd."
            );
        }
        return Ok(());
    }
    match (trainer.decay_ratio, trainer.decay_steps) {
        (Some(ratio), Some(steps)) => anyhow::bail!(
            "trainer.decay_ratio ({ratio}) and trainer.decay_steps ({steps}) both set the wsd \
             decay span. Set one of them."
        ),
        (Some(ratio), None) if ratio.is_nan() || ratio <= 0.0 || ratio > 1.0 => anyhow::bail!(
            "trainer.decay_ratio ({ratio}) must be greater than 0.0 and at most 1.0: it names \
             the fraction of the run's steps the wsd schedule decays across."
        ),
        (None, Some(0)) => anyhow::bail!(
            "trainer.decay_steps is 0. Set a positive step count for the wsd decay span, or \
             remove it to use decay_ratio (default {DEFAULT_DECAY_RATIO})."
        ),
        (None, Some(steps)) if trainer.max_steps > 0 => {
            let warmup = trainer.warmup_steps.unwrap_or(0);
            if warmup.saturating_add(steps) > trainer.max_steps {
                anyhow::bail!(
                    "trainer.warmup_steps ({warmup}) plus trainer.decay_steps ({steps}) exceed \
                     trainer.max_steps ({}). Lower decay_steps or warmup_steps.",
                    trainer.max_steps
                );
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LitConfig;

    const BASE_LR: f64 = 3e-4;
    const TOTAL: usize = 110;

    fn trainer(schedule: LrSchedule, warmup_steps: usize, min_lr_ratio: f64) -> TrainerConfig {
        let mut trainer = LitConfig::nano_baseline().trainer;
        trainer.learning_rate = BASE_LR;
        trainer.max_steps = TOTAL;
        trainer.warmup_steps = Some(warmup_steps);
        trainer.warmup_ratio = None;
        trainer.lr_schedule = schedule;
        trainer.min_lr_ratio = min_lr_ratio;
        trainer
    }

    fn plan(config: &TrainerConfig) -> LrPlan {
        LrPlan::new(config, 10_000).expect("plan")
    }

    /// Frozen copy of the multiplier oxidizr applied before `LrPlan`.
    fn reference_multiplier(
        schedule: LrSchedule,
        step: usize,
        warmup_steps: usize,
        total_steps: usize,
        min_lr_ratio: f64,
    ) -> f64 {
        if schedule == LrSchedule::Constant || step <= warmup_steps {
            return 1.0;
        }
        if total_steps <= warmup_steps {
            return min_lr_ratio;
        }
        let progress =
            ((step - warmup_steps) as f64 / (total_steps - warmup_steps) as f64).min(1.0);
        let shape = match schedule {
            LrSchedule::Cosine => 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()),
            LrSchedule::Linear => 1.0 - progress,
            LrSchedule::Constant | LrSchedule::Wsd => 1.0,
        };
        min_lr_ratio + (1.0 - min_lr_ratio) * shape
    }

    /// Frozen copy of the learning rate the trainer applied before `LrPlan`.
    fn reference_lr(
        schedule: LrSchedule,
        step: usize,
        warmup_steps: usize,
        total_steps: usize,
        min_lr_ratio: f64,
    ) -> f64 {
        if warmup_steps > 0 && step < warmup_steps {
            (BASE_LR * ((step + 1) as f64 / warmup_steps as f64)).max(1e-7)
        } else {
            BASE_LR * reference_multiplier(schedule, step, warmup_steps, total_steps, min_lr_ratio)
        }
    }

    #[test]
    fn boostr_backed_plan_is_bitwise_equal_to_the_reference() {
        for schedule in [LrSchedule::Constant, LrSchedule::Cosine, LrSchedule::Linear] {
            for warmup in [0, 10] {
                for min_ratio in [0.0, 0.1, 0.37] {
                    let plan = plan(&trainer(schedule, warmup, min_ratio));
                    for step in 0..=TOTAL + 5 {
                        let want = reference_lr(schedule, step, warmup, TOTAL, min_ratio);
                        let got = plan.lr_at(step);
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "{schedule:?} warmup={warmup} min={min_ratio} step={step}: \
                             got {got:e}, want {want:e}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_run_shorter_than_warmup_matches_the_reference_floor() {
        let config = trainer(LrSchedule::Cosine, 120, 0.1);
        let plan = plan(&config);
        for step in 0..=TOTAL + 20 {
            let want = reference_lr(LrSchedule::Cosine, step, 120, TOTAL, 0.1);
            assert_eq!(plan.lr_at(step).to_bits(), want.to_bits(), "step={step}");
        }
    }

    #[test]
    fn warmup_ratio_and_max_steps_resolve_the_span() {
        let mut config = trainer(LrSchedule::Cosine, 0, 0.0);
        config.warmup_steps = None;
        config.warmup_ratio = Some(0.1);
        config.max_steps = 0;
        let plan = LrPlan::new(&config, 95).expect("plan");
        assert_eq!(plan.warmup_steps(), 10);
        assert_eq!(plan.total_steps(), 95);
        config.max_steps = 40;
        assert_eq!(LrPlan::new(&config, 95).expect("plan").total_steps(), 40);
    }

    fn wsd(decay_steps: Option<usize>, decay_ratio: Option<f64>, shape: LrDecayShape) -> LrPlan {
        let mut config = trainer(LrSchedule::Wsd, 10, 0.1);
        config.decay_steps = decay_steps;
        config.decay_ratio = decay_ratio;
        config.decay_shape = Some(shape);
        plan(&config)
    }

    #[test]
    fn wsd_phase_boundaries() {
        // warmup 0..10, stable 10..90, decay 90..110, floor from 110.
        for shape in [
            LrDecayShape::Linear,
            LrDecayShape::Cosine,
            LrDecayShape::Sqrt,
        ] {
            let plan = wsd(Some(20), None, shape);
            assert_eq!(plan.lr_at(0), BASE_LR * (1.0 / 10.0), "{shape:?}");
            assert_eq!(plan.lr_at(9), BASE_LR, "{shape:?} last warmup step");
            assert_eq!(plan.lr_at(10), BASE_LR, "{shape:?} first stable step");
            assert_eq!(plan.lr_at(89), BASE_LR, "{shape:?} last stable step");
            assert_eq!(plan.lr_at(90), BASE_LR, "{shape:?} decay progress 0");
            assert!(plan.lr_at(91) < BASE_LR, "{shape:?} decay starts");
            assert!(plan.lr_at(109) > BASE_LR * 0.1, "{shape:?} last decay step");
            assert_eq!(plan.lr_at(TOTAL), BASE_LR * 0.1, "{shape:?} floor");
            assert_eq!(plan.lr_at(TOTAL + 50), BASE_LR * 0.1, "{shape:?} past end");
        }
    }

    #[test]
    fn wsd_decay_midpoints_follow_the_shape() {
        let floor = 0.1;
        let at_mid = |shape| wsd(Some(20), None, shape).lr_at(100) / BASE_LR;
        let linear = floor + (1.0 - floor) * 0.5;
        let sqrt = floor + (1.0 - floor) * (1.0 - 0.5f64.sqrt());
        assert!((at_mid(LrDecayShape::Linear) - linear).abs() < 1e-12);
        assert!((at_mid(LrDecayShape::Cosine) - linear).abs() < 1e-12);
        assert!((at_mid(LrDecayShape::Sqrt) - sqrt).abs() < 1e-12);
    }

    #[test]
    fn wsd_decay_ratio_defaults_to_a_tenth_of_the_run() {
        // ceil(110 * 0.1) = 11 decay steps: decay starts at step 99.
        let plan = wsd(None, None, LrDecayShape::Linear);
        assert_eq!(plan.lr_at(99), BASE_LR);
        assert!(plan.lr_at(100) < BASE_LR);
        let half = wsd(None, Some(0.5), LrDecayShape::Linear);
        assert_eq!(half.lr_at(55), BASE_LR);
        assert!(half.lr_at(56) < BASE_LR);
    }

    #[test]
    fn wsd_decay_that_overlaps_warmup_is_an_error() {
        let mut config = trainer(LrSchedule::Wsd, 10, 0.1);
        config.decay_steps = Some(101);
        let err = LrPlan::new(&config, 10_000)
            .err()
            .expect("overlap rejected");
        assert!(err.to_string().contains("trainer.decay_steps"), "{err}");
    }

    #[test]
    fn constant_does_not_need_total_steps() {
        assert!(!LrSchedule::Constant.needs_total_steps());
        assert!(LrSchedule::Cosine.needs_total_steps());
        assert!(LrSchedule::Linear.needs_total_steps());
        assert!(LrSchedule::Wsd.needs_total_steps());
    }

    #[test]
    fn wsd_keys_parse_from_yaml() {
        let yaml = "lr_schedule: wsd\ndecay_steps: 20\ndecay_shape: sqrt\n";
        #[derive(Deserialize)]
        struct Keys {
            lr_schedule: LrSchedule,
            decay_steps: Option<usize>,
            decay_shape: Option<LrDecayShape>,
        }
        let keys: Keys = serde_yaml::from_str(yaml).expect("parse");
        assert_eq!(keys.lr_schedule, LrSchedule::Wsd);
        assert_eq!(keys.decay_steps, Some(20));
        assert_eq!(keys.decay_shape, Some(LrDecayShape::Sqrt));
    }

    fn validate_err(edit: impl FnOnce(&mut TrainerConfig)) -> String {
        let mut config = trainer(LrSchedule::Wsd, 10, 0.1);
        edit(&mut config);
        validate(&config).expect_err("rejected").to_string()
    }

    #[test]
    fn validate_rejects_bad_wsd_keys() {
        let err = validate_err(|c| {
            c.decay_ratio = Some(0.2);
            c.decay_steps = Some(5);
        });
        assert!(err.contains("Set one of them"), "{err}");
        let err = validate_err(|c| c.decay_ratio = Some(0.0));
        assert!(err.contains("trainer.decay_ratio"), "{err}");
        let err = validate_err(|c| c.decay_ratio = Some(f64::NAN));
        assert!(err.contains("trainer.decay_ratio"), "{err}");
        let err = validate_err(|c| c.decay_steps = Some(0));
        assert!(err.contains("trainer.decay_steps is 0"), "{err}");
        let err = validate_err(|c| c.decay_steps = Some(101));
        assert!(err.contains("exceed trainer.max_steps"), "{err}");
        let err = validate_err(|c| {
            c.lr_schedule = LrSchedule::Cosine;
            c.decay_shape = Some(LrDecayShape::Sqrt);
        });
        assert!(err.contains("apply only to lr_schedule: wsd"), "{err}");
    }

    #[test]
    fn validate_accepts_wsd_defaults_and_explicit_spans() {
        let mut config = trainer(LrSchedule::Wsd, 10, 0.1);
        validate(&config).expect("defaults");
        config.decay_steps = Some(100);
        validate(&config).expect("warmup plus decay fills the run");
    }
}
