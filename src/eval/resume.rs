//! The eval keys of a checkpoint meta: what a save records and what a resume reads back.

use crate::config::CheckpointMeta;
use crate::prelude::{Error, Result};

/// The eval keys a checkpoint meta records. All `None` for a run without eval.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EvalCheckpoint {
    /// Rows held out of training. `None` when training reads every row.
    pub holdout_rows: Option<usize>,
    pub best_eval_loss: Option<f64>,
    pub evals_since_best: Option<usize>,
}

/// The stop counters a resumed run continues from.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EvalResume {
    pub best_eval_loss: Option<f64>,
    pub evals_since_best: usize,
}

/// Reads the eval counters of the meta at `meta_path` for a run at `start_step`.
///
/// `holdout_rows` is this run's held-out row count, 0 without a holdout. A
/// run at step 0 and a resume without a meta start fresh counters.
///
/// # Errors
/// Returns an error when the meta is unreadable, or when it records another
/// held-out row count. The held-out rows decide which rows training reads,
/// so a saved data position means other rows under another count.
pub fn read_eval_resume(
    meta_path: Option<&str>,
    start_step: usize,
    holdout_rows: usize,
) -> Result<EvalResume> {
    let Some(path) = meta_path.filter(|_| start_step > 0) else {
        return Ok(EvalResume::default());
    };
    let meta = CheckpointMeta::load(path).map_err(|e| Error::TrainingError {
        reason: format!(
            "checkpoint meta '{path}' was not read: {e}. Restore the file to resume with its \
             eval state."
        ),
    })?;
    let saved = meta.holdout_rows.unwrap_or(0);
    if saved != holdout_rows {
        return Err(Error::TrainingError {
            reason: format!(
                "checkpoint meta '{path}' records {saved} held-out rows, and this run holds out \
                 {holdout_rows}. The held-out rows decide which rows training reads, so the \
                 saved data position does not carry over. Set trainer.eval to hold out {saved} \
                 rows, or start a fresh run."
            ),
        });
    }
    Ok(EvalResume {
        best_eval_loss: meta.best_eval_loss,
        evals_since_best: meta.evals_since_best.unwrap_or(0),
    })
}
