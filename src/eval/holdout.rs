//! The split of the training rows into a training range and a held-out eval range.

use super::config::EvalConfig;
use crate::prelude::{Error, Result};

/// The rows training reads and the rows held out for eval.
///
/// Training reads stream rows `[0, train_rows)`. Held-out evals read
/// `[train_rows, train_rows + holdout_rows)`. An eval on `trainer.eval.data_file`
/// holds out no rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HoldoutPlan {
    pub train_rows: usize,
    pub holdout_rows: usize,
}

impl HoldoutPlan {
    /// The plan of a run without held-out rows: training reads all `total_rows`.
    pub fn none(total_rows: usize) -> Self {
        Self {
            train_rows: total_rows,
            holdout_rows: 0,
        }
    }

    /// Splits `holdout_rows` or `holdout_fraction` off the end of `total_rows`.
    ///
    /// # Errors
    /// Returns an error when the held-out rows or the training rows cannot fill
    /// one batch of `batch_size` rows. The error names the key to change.
    pub fn for_config(eval: &EvalConfig, total_rows: usize, batch_size: usize) -> Result<Self> {
        if eval.data_file.is_some() {
            return Ok(Self::none(total_rows));
        }
        let (holdout_rows, key) = match (eval.holdout_rows, eval.holdout_fraction) {
            (Some(rows), _) => (rows, "trainer.eval.holdout_rows"),
            (None, Some(fraction)) => (
                (total_rows as f64 * fraction).floor() as usize,
                "trainer.eval.holdout_fraction",
            ),
            (None, None) => (0, "trainer.eval.holdout_rows"),
        };
        if holdout_rows < batch_size {
            return Err(Error::DataError {
                reason: format!(
                    "{key} holds out {holdout_rows} of {total_rows} rows, fewer than one eval \
                     batch of trainer.batch_size {batch_size} rows. Raise {key}."
                ),
            });
        }
        let train_rows = total_rows.saturating_sub(holdout_rows);
        if train_rows < batch_size {
            return Err(Error::DataError {
                reason: format!(
                    "{key} holds out {holdout_rows} of {total_rows} rows and leaves \
                     {train_rows} training rows, fewer than one batch of trainer.batch_size \
                     {batch_size} rows. Lower {key}, or use more data."
                ),
            });
        }
        Ok(Self {
            train_rows,
            holdout_rows,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(yaml: &str) -> EvalConfig {
        serde_yaml::from_str(yaml).expect("eval config")
    }

    #[test]
    fn rows_and_fractions_split_off_the_end() {
        let rows = HoldoutPlan::for_config(&eval("holdout_rows: 30"), 100, 4).expect("rows");
        assert_eq!(
            rows,
            HoldoutPlan {
                train_rows: 70,
                holdout_rows: 30
            }
        );
        let fraction =
            HoldoutPlan::for_config(&eval("holdout_fraction: 0.057"), 100, 4).expect("fraction");
        assert_eq!(fraction.holdout_rows, 5);
        assert_eq!(fraction.train_rows, 95);
        let file = HoldoutPlan::for_config(&eval("data_file: e.bin"), 100, 4).expect("file");
        assert_eq!(file, HoldoutPlan::none(100));
    }

    #[test]
    fn too_few_rows_on_either_side_is_an_error() {
        let err = HoldoutPlan::for_config(&eval("holdout_fraction: 0.01"), 100, 4)
            .expect_err("1 row holds no batch of 4");
        assert!(
            err.to_string().contains("trainer.eval.holdout_fraction"),
            "{err}"
        );
        let err = HoldoutPlan::for_config(&eval("holdout_rows: 98"), 100, 4)
            .expect_err("2 training rows hold no batch of 4");
        assert!(err.to_string().contains("leaves 2 training rows"), "{err}");
        let err = HoldoutPlan::for_config(&eval("holdout_rows: 120"), 100, 4)
            .expect_err("more rows than the data");
        assert!(err.to_string().contains("leaves 0 training rows"), "{err}");
    }
}
