//! The split of the training rows into a training range and a held-out eval range.
//!
//! Over several data sources, `holdout_fraction` applies to each source and
//! takes rows off the end of each. The eval reads the held-out ranges in
//! source order. `holdout_rows` names one range, so it needs one source.

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
                fraction_rows(total_rows, fraction),
                "trainer.eval.holdout_fraction",
            ),
            (None, None) => (0, "trainer.eval.holdout_rows"),
        };
        check_split(key, holdout_rows, total_rows, batch_size)
    }
}

/// `fraction` of `rows`, rounded down.
fn fraction_rows(rows: usize, fraction: f64) -> usize {
    (rows as f64 * fraction).floor() as usize
}

/// The plan of `holdout_rows` off `total_rows`, after checking both sides hold a batch.
fn check_split(
    key: &str,
    holdout_rows: usize,
    total_rows: usize,
    batch_size: usize,
) -> Result<HoldoutPlan> {
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
    Ok(HoldoutPlan {
        train_rows,
        holdout_rows,
    })
}

/// A [`HoldoutPlan`] with the global row ranges each side reads.
///
/// The training and eval ranges are disjoint. Each list ascends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HoldoutSplit {
    /// Row totals of both sides.
    pub plan: HoldoutPlan,
    /// `(start, end)` global rows training reads.
    pub train_ranges: Vec<(usize, usize)>,
    /// `(start, end)` global rows the eval reads. Empty without held-out rows.
    pub eval_ranges: Vec<(usize, usize)>,
}

impl HoldoutSplit {
    /// Splits held-out rows off the end of each source of `source_rows` rows.
    ///
    /// One source splits as [`HoldoutPlan::for_config`] does. Several sources
    /// take `holdout_fraction` of each source, rounded down per source.
    ///
    /// # Errors
    /// Returns the errors of [`HoldoutPlan::for_config`] on the totals, and an
    /// error for `holdout_rows` over more than one source.
    pub fn for_sources(
        eval: &EvalConfig,
        source_rows: &[usize],
        batch_size: usize,
    ) -> Result<Self> {
        let total: usize = source_rows.iter().sum();
        if let [rows] = source_rows {
            let plan = HoldoutPlan::for_config(eval, *rows, batch_size)?;
            return Ok(Self::of_plan(plan));
        }
        if eval.data_file.is_some() {
            return Ok(Self::of_plan(HoldoutPlan::none(total)));
        }
        let fraction = match (eval.holdout_rows, eval.holdout_fraction) {
            (Some(_), _) => {
                return Err(Error::DataError {
                    reason: format!(
                        "trainer.eval.holdout_rows splits rows off one data source, and the run \
                         reads {} sources. Use trainer.eval.holdout_fraction: it holds out that \
                         fraction of each source.",
                        source_rows.len()
                    ),
                })
            }
            (None, fraction) => fraction.unwrap_or(0.0),
        };
        let (mut train_ranges, mut eval_ranges) = (Vec::new(), Vec::new());
        let (mut first, mut held) = (0, 0);
        for &rows in source_rows {
            let out = fraction_rows(rows, fraction);
            let cut = first + rows - out;
            if cut > first {
                train_ranges.push((first, cut));
            }
            if out > 0 {
                eval_ranges.push((cut, first + rows));
            }
            held += out;
            first += rows;
        }
        let plan = check_split("trainer.eval.holdout_fraction", held, total, batch_size)?;
        Ok(Self {
            plan,
            train_ranges,
            eval_ranges,
        })
    }

    /// The ranges of a one-range `plan`: `[0, train_rows)` and the rows after it.
    fn of_plan(plan: HoldoutPlan) -> Self {
        let end = plan.train_rows + plan.holdout_rows;
        let train_ranges = if plan.train_rows > 0 {
            vec![(0, plan.train_rows)]
        } else {
            Vec::new()
        };
        let eval_ranges = if plan.holdout_rows > 0 {
            vec![(plan.train_rows, end)]
        } else {
            Vec::new()
        };
        Self {
            plan,
            train_ranges,
            eval_ranges,
        }
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
    fn several_sources_hold_out_the_end_of_each() {
        let split = HoldoutSplit::for_sources(&eval("holdout_fraction: 0.25"), &[40, 10, 21], 4)
            .expect("split");
        assert_eq!(split.eval_ranges, vec![(30, 40), (48, 50), (66, 71)]);
        assert_eq!(split.train_ranges, vec![(0, 30), (40, 48), (50, 66)]);
        assert_eq!(split.plan.holdout_rows, 17);
        assert_eq!(split.plan.train_rows, 54);
        let train: usize = split.train_ranges.iter().map(|(s, e)| e - s).sum();
        assert_eq!(train, split.plan.train_rows);
        for &(es, ee) in &split.eval_ranges {
            for &(ts, te) in &split.train_ranges {
                assert!(
                    ee <= ts || te <= es,
                    "eval [{es}, {ee}) overlaps train [{ts}, {te})"
                );
            }
        }
    }

    #[test]
    fn one_source_splits_as_the_plan_does() {
        let split = HoldoutSplit::for_sources(&eval("holdout_rows: 30"), &[100], 4).expect("split");
        assert_eq!(split.train_ranges, vec![(0, 70)]);
        assert_eq!(split.eval_ranges, vec![(70, 100)]);
        let file =
            HoldoutSplit::for_sources(&eval("data_file: e.bin"), &[60, 40], 4).expect("file");
        assert_eq!(file.plan, HoldoutPlan::none(100));
        assert!(file.eval_ranges.is_empty());
    }

    #[test]
    fn holdout_rows_over_several_sources_is_an_error() {
        let err = HoldoutSplit::for_sources(&eval("holdout_rows: 8"), &[40, 40], 4)
            .expect_err("holdout_rows needs one source");
        let text = err.to_string();
        assert!(text.contains("trainer.eval.holdout_fraction"), "{text}");
        assert!(text.contains("reads 2 sources"), "{text}");
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
