//! Files a training run writes to its run directory.
//!
//! - `run.json`: one [`RunRecord`], written once the run ends, in both output modes.
//! - `steps.json`: one [`StepRecord`] JSON object per line, appended at each log interval.
//! - `evals.json`: one [`EvalRecord`] JSON object per line, appended at each eval.
//!   Absent for a run without `trainer.eval`.

use crate::eval::EvalOutcome;
use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

/// File name of the run record inside the run directory.
pub const RUN_RECORD_FILE: &str = "run.json";
/// File name of the per-step log inside the run directory.
pub const STEPS_FILE: &str = "steps.json";
/// File name of the per-eval log inside the run directory.
pub const EVALS_FILE: &str = "evals.json";

/// How a run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunStatus {
    Completed,
    Failed,
}

/// The contents of `run.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    pub run_dir: String,
    pub status: RunStatus,
    /// The error chain of a failed run. `None` for a completed run.
    pub error: Option<String>,
    /// RFC 3339 local time when the command started.
    pub started_at: String,
    /// RFC 3339 local time when the command ended.
    pub finished_at: String,
    /// Completed optimizer steps when the run ended, resumed steps included.
    pub final_step: usize,
    /// Loss of the last completed step. `None` when no step ran.
    pub last_loss: Option<f64>,
    /// Runtime name, such as `cpu`.
    pub device: String,
    /// Token count of the training data. `None` when setup stopped before the data loaded.
    pub dataset_size: Option<usize>,
    /// Model and trainer summary. `None` when setup stopped before it was known.
    pub config: Option<serde_json::Value>,
    /// Lowest eval loss of the run. Absent without `trainer.eval`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_eval_loss: Option<f64>,
    /// Evals this run directory recorded. Absent without `trainer.eval`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evals: Option<usize>,
    /// True when early stopping ended the run before its last step.
    #[serde(default, skip_serializing_if = "is_false")]
    pub stopped_early: bool,
    /// Why early stopping ended the run. Absent when it ran every step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

fn is_false(value: &bool) -> bool {
    !*value
}

impl RunRecord {
    /// Path of `run.json` inside `run_dir`.
    pub fn path_in(run_dir: &Path) -> PathBuf {
        run_dir.join(RUN_RECORD_FILE)
    }

    /// Writes `run.json` into `run_dir`.
    pub fn write(&self, run_dir: &Path) -> anyhow::Result<()> {
        let path = Self::path_in(run_dir);
        let json = serde_json::to_string_pretty(self).context("failed to serialize run.json")?;
        boostr::format::write_atomic(&path, |w| w.write_all(json.as_bytes())).map_err(|e| {
            anyhow::anyhow!(
                "failed to write run record '{}': {e}. Check that the run directory is writable.",
                path.display()
            )
        })
    }

    /// Reads `run.json` from `run_dir`.
    pub fn load(run_dir: &Path) -> anyhow::Result<Self> {
        let path = Self::path_in(run_dir);
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read run record '{}'", path.display()))?;
        serde_json::from_str(&content)
            .with_context(|| format!("failed to parse run record '{}'", path.display()))
    }
}

/// What the train command knows about a run so far.
///
/// The command fills the fields as setup and training advance. [`RunProgress::finish`]
/// turns it into a [`RunRecord`] for the result, so a run that stops early
/// still records what it reached.
#[derive(Debug, Clone)]
pub struct RunProgress {
    pub run_dir: String,
    pub started_at: String,
    pub device: String,
    pub dataset_size: Option<usize>,
    pub config: Option<serde_json::Value>,
    pub final_step: usize,
    pub last_loss: Option<f64>,
    /// What the evals reached. `None` without `trainer.eval`.
    pub eval: Option<EvalOutcome>,
}

impl RunProgress {
    /// Starts the record of a run in `run_dir` on the runtime named `device`.
    pub fn start(run_dir: &str, device: &str) -> Self {
        Self {
            run_dir: run_dir.to_string(),
            started_at: chrono::Local::now().to_rfc3339(),
            device: device.to_string(),
            dataset_size: None,
            config: None,
            final_step: 0,
            last_loss: None,
            eval: None,
        }
    }

    /// The record of a run that ended with `result`.
    pub fn finish(&self, result: &anyhow::Result<()>) -> RunRecord {
        let (status, error) = match result {
            Ok(()) => (RunStatus::Completed, None),
            Err(e) => (RunStatus::Failed, Some(format!("{e:#}"))),
        };
        RunRecord {
            run_dir: self.run_dir.clone(),
            status,
            error,
            started_at: self.started_at.clone(),
            finished_at: chrono::Local::now().to_rfc3339(),
            final_step: self.final_step,
            last_loss: self.last_loss,
            device: self.device.clone(),
            dataset_size: self.dataset_size,
            config: self.config.clone(),
            best_eval_loss: self.eval.as_ref().and_then(|e| e.best_eval_loss),
            evals: self.eval.as_ref().map(|e| e.evals),
            stopped_early: self.eval.as_ref().is_some_and(|e| e.stop_reason.is_some()),
            stop_reason: self.eval.as_ref().and_then(|e| e.stop_reason.clone()),
        }
    }
}

/// One line of `evals.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalRecord {
    /// Completed optimizer steps when the eval ran.
    pub step: usize,
    /// Cross-entropy per scored eval token.
    pub loss: f64,
    pub perplexity: f64,
    pub batches: usize,
    /// Scored eval tokens.
    pub tokens: usize,
    /// Lowest eval loss so far, this one included.
    pub best_loss: Option<f64>,
    /// Evals since the best loss.
    pub since_best: usize,
}

impl EvalRecord {
    /// Appends this record as one JSON line to `path`, creating the file when absent.
    pub fn append_to(&self, path: &Path) -> anyhow::Result<()> {
        append_line(self, path, "eval log")
    }
}

/// Appends `value` as one JSON line to the `what` file at `path`.
fn append_line<T: Serialize>(value: &T, path: &Path, what: &str) -> anyhow::Result<()> {
    let line = serde_json::to_string(value)
        .with_context(|| format!("failed to serialize a {what} line"))?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open {what} '{}'", path.display()))?;
    writeln!(file, "{line}")
        .with_context(|| format!("failed to append to {what} '{}'", path.display()))
}

/// One line of `steps.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    pub step: usize,
    pub loss: f64,
    pub grad_norm: f64,
    pub learning_rate: f64,
    pub epoch: f64,
}

impl StepRecord {
    /// Appends this record as one JSON line to `path`, creating the file when absent.
    pub fn append_to(&self, path: &Path) -> anyhow::Result<()> {
        append_line(self, path, "step log")
    }
}

/// The process exit code for a command result: 0 on success, 1 on error.
pub fn exit_code(result: &anyhow::Result<()>) -> i32 {
    match result {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failed_result_records_the_error_chain() {
        let mut progress = RunProgress::start("runs/x", "cpu");
        progress.final_step = 3;
        progress.last_loss = Some(2.5);
        let result: anyhow::Result<()> =
            Err(anyhow::anyhow!("root cause").context("training stopped"));

        let record = progress.finish(&result);

        assert_eq!(record.status, RunStatus::Failed);
        assert_eq!(
            record.error.as_deref(),
            Some("training stopped: root cause")
        );
        assert_eq!(record.final_step, 3);
        assert_eq!(record.last_loss, Some(2.5));
        assert_eq!(exit_code(&result), 1);
    }

    #[test]
    fn a_completed_result_records_no_error_and_exit_code_zero() {
        let result: anyhow::Result<()> = Ok(());
        let record = RunProgress::start("runs/x", "cpu").finish(&result);

        assert_eq!(record.status, RunStatus::Completed);
        assert_eq!(record.error, None);
        let json = serde_json::to_value(&record).expect("serialize");
        assert_eq!(json["status"], "completed");
        assert_eq!(exit_code(&result), 0);
        for key in ["best_eval_loss", "evals", "stopped_early", "stop_reason"] {
            assert!(json.get(key).is_none(), "{key} without an eval");
        }
    }

    #[test]
    fn an_early_stop_completes_with_its_reason_and_exit_code_zero() {
        let mut progress = RunProgress::start("runs/x", "cpu");
        progress.eval = Some(EvalOutcome {
            best_eval_loss: Some(1.25),
            evals: 4,
            stop_reason: Some("patience spent".to_string()),
        });
        let result: anyhow::Result<()> = Ok(());
        let json = serde_json::to_value(progress.finish(&result)).expect("serialize");

        assert_eq!(json["status"], "completed");
        assert_eq!(json["best_eval_loss"], 1.25);
        assert_eq!(json["evals"], 4);
        assert_eq!(json["stopped_early"], true);
        assert_eq!(json["stop_reason"], "patience spent");
        assert_eq!(exit_code(&result), 0);
    }
}
