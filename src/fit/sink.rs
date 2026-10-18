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

/// A checkpoint event of the fit loop.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointNote {
    /// The interval checkpoint labelled `step` is on disk.
    Saved { step: usize },
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
}

/// Receives the events of one fit loop, in order.
///
/// The loop calls `on_start` once, then `on_step` for every step, with
/// `on_checkpoint` after each interval save. It ends with
/// `on_checkpoint(Final)` and `on_finish`, or with `on_error` when a step
/// returns an error. An error from a hook stops the loop with that error.
pub trait StepSink {
    fn on_start(&mut self, start: &FitStart) -> Result<()>;
    fn on_step(&mut self, report: &StepReport) -> Result<()>;
    fn on_checkpoint(&mut self, note: &CheckpointNote) -> Result<()>;
    /// The training step with 1-based number `step` returned `error`.
    ///
    /// The loop returns `error` right after this call, so the hook reports only.
    fn on_error(&mut self, step: usize, error: &Error);
    fn on_finish(&mut self, summary: &FitSummary) -> Result<()>;
}
