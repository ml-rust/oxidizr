//! The scalar log of a run: TensorBoard events under the `tensorboard` feature.
//!
//! Without the feature every call does nothing. A write error prints one
//! warning and closes the log. Training never stops for it.

use crate::fit::{EvalReport, StepReport};
use std::path::Path;

/// TensorBoard tags of the scalars a run logs.
pub mod tags {
    pub const LOSS: &str = "train/loss";
    pub const LEARNING_RATE: &str = "train/learning_rate";
    pub const GRAD_NORM: &str = "train/grad_norm";
    pub const TOKENS_PER_SEC: &str = "train/tokens_per_sec";
    pub const MFU: &str = "train/mfu";
    pub const EVAL_LOSS: &str = "eval/loss";
    pub const EVAL_PERPLEXITY: &str = "eval/perplexity";
}

/// The scalar log of one fit session. Rank 0 alone opens one.
#[derive(Default)]
pub struct ScalarLog {
    #[cfg(feature = "tensorboard")]
    writer: Option<super::tensorboard::EventWriter>,
}

impl ScalarLog {
    /// A log that writes nothing.
    pub fn off() -> Self {
        Self::default()
    }

    /// A new event file in `dir`.
    ///
    /// An error prints a warning and returns a log that writes nothing.
    #[cfg(feature = "tensorboard")]
    pub fn open(dir: &Path) -> Self {
        match super::tensorboard::EventWriter::create(dir) {
            Ok(writer) => Self {
                writer: Some(writer),
            },
            Err(e) => {
                eprintln!(
                    "Warning: TensorBoard event file in '{}' was not created: {e}. Training \
                     continues without TensorBoard events.",
                    dir.display()
                );
                Self::off()
            }
        }
    }

    /// A log that writes nothing: this build has no `tensorboard` feature.
    #[cfg(not(feature = "tensorboard"))]
    pub fn open(_dir: &Path) -> Self {
        Self::off()
    }

    /// The scalars of the logged step `report`.
    pub fn step(&mut self, report: &StepReport) {
        let mut values = vec![
            (tags::LOSS, report.loss),
            (tags::LEARNING_RATE, report.learning_rate),
            (tags::GRAD_NORM, report.grad_norm),
            (tags::TOKENS_PER_SEC, report.tokens_per_sec),
        ];
        if let Some(mfu) = report.mfu {
            values.push((tags::MFU, mfu));
        }
        self.write(report.step, &values);
    }

    /// The scalars of the eval after `step` completed steps.
    pub fn eval(&mut self, step: usize, report: &EvalReport) {
        self.write(
            step,
            &[
                (tags::EVAL_LOSS, report.loss),
                (tags::EVAL_PERPLEXITY, report.perplexity),
            ],
        );
    }

    #[cfg(feature = "tensorboard")]
    fn write(&mut self, step: usize, values: &[(&str, f64)]) {
        let Some(writer) = self.writer.as_mut() else {
            return;
        };
        if let Err(e) = writer.scalars(step as u64, values) {
            eprintln!(
                "Warning: TensorBoard event file '{}' was not written: {e}. Training \
                 continues without TensorBoard events.",
                writer.path().display()
            );
            self.writer = None;
        }
    }

    #[cfg(not(feature = "tensorboard"))]
    fn write(&mut self, _step: usize, _values: &[(&str, f64)]) {}
}
