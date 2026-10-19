//! The eval state a trainer carries: its loader, its schedule and its stop counters.

use super::config::EvalConfig;
use super::holdout::HoldoutSplit;
use super::resume::{EvalCheckpoint, EvalResume};
use super::stopper::EarlyStopper;
use crate::data::{LitDataLoader, RowOrder};
use crate::fit::EvalReport;
use crate::prelude::{DType, Error, Result};
use crate::run_record::EvalRecord;
use boostr::runtime::Runtime;
use std::path::PathBuf;

/// What a run's evals reached, for `run.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalOutcome {
    /// Lowest eval loss of the run, resumed evals included.
    pub best_eval_loss: Option<f64>,
    /// Evals this session ran.
    pub evals: usize,
    /// Why the run stopped early. `None` when it ran every step.
    pub stop_reason: Option<String>,
}

/// The eval of one run.
///
/// It owns its own loader, so an eval never moves the training loader or its
/// position, and draws from no RNG that training uses.
pub struct Evaluator<R: Runtime<DType = DType>> {
    pub(super) loader: LitDataLoader<R>,
    /// Batches per eval, capped at the whole batches of the eval rows.
    pub(super) batches: usize,
    interval: usize,
    holdout_rows: usize,
    pub(super) stopper: EarlyStopper,
    /// `evals.json` beside the trainer's `steps.json`. `None` until attached.
    pub(super) evals_path: Option<PathBuf>,
    pub(super) evals_run: usize,
    pub(super) stop_reason: Option<String>,
}

impl<R: Runtime<DType = DType>> Evaluator<R> {
    /// The eval of `config` over `loader`.
    ///
    /// `loader` must read the same sources and loss mask as training does.
    /// Under held-out rows it narrows to `split.eval_ranges`, read in order.
    /// An eval on `trainer.eval.data_file` reads every row of that file. Rows
    /// go in stream order, so every eval sees the same batches.
    ///
    /// # Errors
    /// Returns an error when the eval rows hold no whole batch.
    pub fn new(
        config: &EvalConfig,
        split: &HoldoutSplit,
        loader: LitDataLoader<R>,
    ) -> Result<Self> {
        let plan = split.plan;
        let loader = if plan.holdout_rows > 0 {
            loader.with_row_ranges(&split.eval_ranges)?
        } else {
            loader
        };
        let loader = loader.with_row_order(RowOrder::Stream);
        let layout = loader.row_layout().map_err(|e| Error::DataError {
            reason: format!("eval rows: {e}"),
        })?;
        Ok(Self {
            loader,
            batches: config.batches.min(layout.batches_per_epoch()),
            interval: config.interval,
            holdout_rows: plan.holdout_rows,
            stopper: EarlyStopper::for_config(config.early_stopping.as_ref()),
            evals_path: None,
            evals_run: 0,
            stop_reason: None,
        })
    }

    /// Continues the stop counters a checkpoint recorded.
    ///
    /// A run whose restored counters already spent the patience stops before its first step.
    pub fn resumed(mut self, resume: EvalResume) -> Self {
        self.stopper = self
            .stopper
            .restored(resume.best_eval_loss, resume.evals_since_best);
        if self.stopper.patience_spent() {
            self.stop_reason = Some(self.stop_text());
        }
        self
    }

    /// Batches each eval scores.
    pub fn batches(&self) -> usize {
        self.batches
    }

    /// Rows split off the end of the training data. 0 for an eval on its own file.
    pub fn holdout_rows(&self) -> usize {
        self.holdout_rows
    }

    /// The stream rows `(start, end)` each eval reads, end exclusive.
    ///
    /// Over several sources this spans the first held-out range to the end
    /// of the last. [`row_ranges`](Self::row_ranges) gives each range.
    pub fn row_range(&self) -> (usize, usize) {
        self.loader.row_range()
    }

    /// The global row ranges each eval reads, in order.
    pub fn row_ranges(&self) -> &[(usize, usize)] {
        self.loader.row_ranges()
    }

    /// Whether an eval runs once `completed` of `total_steps` steps are done.
    pub fn is_due(&self, completed: usize, total_steps: usize) -> bool {
        completed.is_multiple_of(self.interval) || completed == total_steps
    }

    pub(crate) fn set_evals_path(&mut self, path: PathBuf) {
        self.evals_path = Some(path);
    }

    /// The eval counters a checkpoint records.
    pub fn checkpoint_state(&self) -> EvalCheckpoint {
        EvalCheckpoint {
            holdout_rows: (self.holdout_rows > 0).then_some(self.holdout_rows),
            best_eval_loss: self.stopper.best(),
            evals_since_best: Some(self.stopper.since_best()),
        }
    }

    /// What this run reached, for `run.json`.
    pub fn outcome(&self) -> EvalOutcome {
        EvalOutcome {
            best_eval_loss: self.stopper.best(),
            evals: self.evals_run,
            stop_reason: self.stop_reason.clone(),
        }
    }

    /// Why the run stops. `None` while the patience lasts.
    pub fn stop_reason(&self) -> Option<&str> {
        self.stop_reason.as_deref()
    }

    pub(super) fn stop_text(&self) -> String {
        let best = self
            .stopper
            .best()
            .map_or_else(|| "none".to_string(), |b| format!("{b:.4}"));
        format!(
            "no eval loss beat the best ({best}) by more than \
             trainer.eval.early_stopping.min_delta {} in {} evals",
            self.stopper.min_delta(),
            self.stopper.since_best()
        )
    }

    /// Appends the `evals.json` line of the eval after step `step`.
    ///
    /// A write error prints a warning and never stops the run.
    pub(super) fn record(&self, step: usize, report: &EvalReport) {
        let Some(path) = &self.evals_path else {
            return;
        };
        let record = EvalRecord {
            step,
            loss: report.loss,
            perplexity: report.perplexity,
            batches: report.batches,
            tokens: report.tokens,
            best_loss: report.best_loss,
            since_best: report.since_best,
        };
        if let Err(e) = record.append_to(path) {
            eprintln!("Warning: {e}");
        }
    }
}
