//! The early-stop rule over a sequence of eval losses.

use super::config::EarlyStoppingConfig;

/// What one eval loss means for the run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Decision {
    /// The loss is the new best.
    Improved,
    /// No improvement. `since_best` counts the evals since the best, this one included.
    NotImproved { since_best: usize },
    /// No improvement for `patience` evals in a row. The run stops.
    Stop { since_best: usize },
}

/// Tracks the best eval loss and the evals since it.
///
/// An eval improves when its loss is below `best - min_delta`. The first
/// finite loss always improves. A non-finite loss never does. Without a
/// patience the stopper tracks the best loss and never stops.
#[derive(Debug, Clone, PartialEq)]
pub struct EarlyStopper {
    patience: Option<usize>,
    min_delta: f64,
    best: Option<f64>,
    since_best: usize,
}

impl EarlyStopper {
    /// A stopper with no eval seen yet.
    pub fn new(patience: Option<usize>, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            best: None,
            since_best: 0,
        }
    }

    /// A stopper for the `trainer.eval.early_stopping` block, or none.
    pub fn for_config(config: Option<&EarlyStoppingConfig>) -> Self {
        Self::new(
            config.map(|c| c.patience),
            config.map_or(0.0, |c| c.min_delta),
        )
    }

    /// Continues from the counters a checkpoint recorded.
    pub fn restored(mut self, best: Option<f64>, since_best: usize) -> Self {
        self.best = best;
        self.since_best = since_best;
        self
    }

    /// Records one eval loss and returns what it means.
    pub fn observe(&mut self, loss: f64) -> Decision {
        let improved =
            loss.is_finite() && self.best.is_none_or(|best| loss < best - self.min_delta);
        if improved {
            self.best = Some(loss);
            self.since_best = 0;
            return Decision::Improved;
        }
        self.since_best += 1;
        let since_best = self.since_best;
        if self.patience_spent() {
            Decision::Stop { since_best }
        } else {
            Decision::NotImproved { since_best }
        }
    }

    /// Whether the evals since the best reached the patience.
    pub fn patience_spent(&self) -> bool {
        self.patience.is_some_and(|p| self.since_best >= p)
    }

    /// The lowest eval loss seen. `None` before the first finite loss.
    pub fn best(&self) -> Option<f64> {
        self.best
    }

    /// Evals since the best loss. Every eval counts until the first finite loss.
    pub fn since_best(&self) -> usize {
        self.since_best
    }

    pub fn min_delta(&self) -> f64 {
        self.min_delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decisions(stopper: &mut EarlyStopper, losses: &[f64]) -> Vec<Decision> {
        losses.iter().map(|l| stopper.observe(*l)).collect()
    }

    #[test]
    fn stop_fires_at_the_eval_that_spends_the_patience() {
        let mut stopper = EarlyStopper::new(Some(2), 0.0);
        let got = decisions(&mut stopper, &[3.0, 2.5, 2.6, 2.4, 2.4, 2.5]);
        assert_eq!(
            got,
            [
                Decision::Improved,
                Decision::Improved,
                Decision::NotImproved { since_best: 1 },
                Decision::Improved,
                Decision::NotImproved { since_best: 1 },
                Decision::Stop { since_best: 2 },
            ]
        );
        assert_eq!(stopper.best(), Some(2.4));
        assert!(stopper.patience_spent());
    }

    #[test]
    fn a_gain_within_min_delta_is_no_improvement() {
        let mut stopper = EarlyStopper::new(Some(3), 0.1);
        let got = decisions(&mut stopper, &[2.0, 1.95, 1.91, 1.85]);
        assert_eq!(
            got,
            [
                Decision::Improved,
                Decision::NotImproved { since_best: 1 },
                Decision::NotImproved { since_best: 2 },
                Decision::Improved,
            ]
        );
        assert_eq!(stopper.best(), Some(1.85));
        assert_eq!(stopper.since_best(), 0);
    }

    #[test]
    fn without_patience_the_stopper_never_stops() {
        let mut stopper = EarlyStopper::for_config(None);
        let got = decisions(&mut stopper, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert!(got.iter().all(|d| !matches!(d, Decision::Stop { .. })));
        assert_eq!(stopper.since_best(), 6);
    }

    #[test]
    fn a_non_finite_loss_never_improves() {
        let mut stopper = EarlyStopper::new(Some(2), 0.0);
        assert_eq!(
            stopper.observe(f64::NAN),
            Decision::NotImproved { since_best: 1 }
        );
        assert_eq!(stopper.best(), None);
        assert_eq!(stopper.observe(1.0), Decision::Improved);
        assert_eq!(
            stopper.observe(f64::INFINITY),
            Decision::NotImproved { since_best: 1 }
        );
        assert_eq!(stopper.observe(f64::NAN), Decision::Stop { since_best: 2 });
    }

    #[test]
    fn a_restored_stopper_continues_the_counters() {
        let losses = [3.0, 2.0, 2.1, 2.2, 1.9, 2.0, 2.0];
        let mut whole = EarlyStopper::new(Some(3), 0.0);
        let reference = decisions(&mut whole, &losses);

        let mut first = EarlyStopper::new(Some(3), 0.0);
        let mut got = decisions(&mut first, &losses[..3]);
        let mut second = EarlyStopper::new(Some(3), 0.0).restored(first.best(), first.since_best());
        got.extend(decisions(&mut second, &losses[3..]));
        assert_eq!(got, reference);
        assert_eq!(second, whole);
    }
}
