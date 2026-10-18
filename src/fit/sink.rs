//! The hooks the fit loop reports to, and the values it passes them.

use crate::prelude::{Error, Result};

/// What the fit loop knows before its first step.
#[derive(Debug, Clone, PartialEq)]
pub struct FitStart {
    /// Steps a resumed checkpoint already completed. 0 for a fresh run.
    pub start_step: usize,
    /// Steps of the whole run, resumed steps included.
    pub total_steps: usize,
    pub warmup_steps: usize,
    pub base_lr: f64,
}

/// One completed optimizer step.
#[derive(Debug, Clone, PartialEq)]
pub struct StepReport {
    /// 1-based count of completed steps, resumed steps included.
    pub step: usize,
    pub loss: f64,
    pub grad_norm: f64,
    /// The learning rate this step applied.
    pub learning_rate: f64,
    /// Fractional epochs of data drawn after this step.
    pub epoch: f64,
    /// Whether this step is a log step. A log step also has a `steps.json` line.
    pub logged: bool,
    /// Samples of this session divided by the session's elapsed seconds.
    pub session_samples_per_sec: f64,
    /// Samples per step divided by the mean time of the recent steps.
    pub recent_samples_per_sec: f64,
    /// Remaining steps times the mean time of the recent steps.
    pub eta_secs: f64,
}

/// One eval at a step boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalReport {
    /// Cross-entropy per scored eval token.
    pub loss: f64,
    /// `exp(loss)`.
    pub perplexity: f64,
    pub batches: usize,
    /// Scored eval tokens.
    pub tokens: usize,
    /// Lowest eval loss so far, this one included. `None` before a finite loss.
    pub best_loss: Option<f64>,
    /// Evals since the best loss. 0 when this eval is the best.
    pub since_best: usize,
}

/// A checkpoint event of the fit loop.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointNote {
    /// The interval checkpoint labelled `step` is on disk.
    Saved { step: usize },
    /// The `best` checkpoint now holds the weights after step `step`.
    Best { step: usize },
    /// The final checkpoint save starts. `path` names its weights file.
    Final { path: String },
}

/// What the fit loop reached when it ended without an error.
#[derive(Debug, Clone, PartialEq)]
pub struct FitSummary {
    /// Completed optimizer steps, resumed steps included.
    pub final_step: usize,
    /// Loss of the last step of this session. `None` when no step ran.
    pub last_loss: Option<f64>,
    /// Gradient norm of the last step of this session. `None` when no step ran.
    pub last_grad_norm: Option<f64>,
    /// Why early stopping ended the run. `None` when it ran every step.
    pub stopped_early: Option<String>,
}

/// Receives the events of one fit loop, in order.
///
/// The loop calls `on_start` once, then `on_step` for every step. After a
/// step it calls `on_eval` when an eval is due, `on_checkpoint(Best)` when
/// that eval wrote the best checkpoint, and `on_checkpoint(Saved)` after an
/// interval save. It ends with `on_checkpoint(Final)` and `on_finish`, or with
/// `on_error` when a step returns an error. An error from a hook stops the
/// loop with that error.
pub trait StepSink {
    fn on_start(&mut self, start: &FitStart) -> Result<()>;
    fn on_step(&mut self, report: &StepReport) -> Result<()>;
    /// The eval after `step` completed steps.
    fn on_eval(&mut self, step: usize, report: &EvalReport) -> Result<()>;
    fn on_checkpoint(&mut self, note: &CheckpointNote) -> Result<()>;
    /// The training step with 1-based number `step` returned `error`.
    ///
    /// The loop returns `error` right after this call, so the hook reports only.
    fn on_error(&mut self, step: usize, error: &Error);
    fn on_finish(&mut self, summary: &FitSummary) -> Result<()>;
}

/// A sink that drops every event. Data-parallel ranks other than 0 report to it.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullSink;

impl StepSink for NullSink {
    fn on_start(&mut self, _start: &FitStart) -> Result<()> {
        Ok(())
    }

    fn on_step(&mut self, _report: &StepReport) -> Result<()> {
        Ok(())
    }

    fn on_eval(&mut self, _step: usize, _report: &EvalReport) -> Result<()> {
        Ok(())
    }

    fn on_checkpoint(&mut self, _note: &CheckpointNote) -> Result<()> {
        Ok(())
    }

    fn on_error(&mut self, _step: usize, _error: &Error) {}

    fn on_finish(&mut self, _summary: &FitSummary) -> Result<()> {
        Ok(())
    }
}
