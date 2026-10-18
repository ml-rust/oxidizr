//! Interactive output: a sticky progress bar with scrolling log lines above it.

use super::sink::{CheckpointNote, EvalReport, FitStart, FitSummary, StepReport, StepSink};
use crate::prelude::{Error, Result};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};

/// Draws the interactive progress of a fit loop.
///
/// A draw error never stops the run, so every print drops its result.
pub struct TuiSink {
    mp: MultiProgress,
    pb: Option<ProgressBar>,
    /// Loss and gradient norm of the last reported step.
    last: (f64, f64),
}

impl TuiSink {
    /// The sink of the interactive mode. It draws to stderr when stderr is a terminal.
    pub fn new() -> Self {
        Self::with_progress(MultiProgress::new())
    }

    /// A sink that draws nothing, for a run that needs the interactive event order only.
    pub fn hidden() -> Self {
        Self::with_progress(MultiProgress::with_draw_target(ProgressDrawTarget::hidden()))
    }

    fn with_progress(mp: MultiProgress) -> Self {
        Self {
            mp,
            pb: None,
            last: (0.0, 0.0),
        }
    }

    fn println(&self, line: impl AsRef<str>) {
        self.mp.println(line).ok();
    }
}

impl Default for TuiSink {
    fn default() -> Self {
        Self::new()
    }
}

impl StepSink for TuiSink {
    fn on_start(&mut self, start: &FitStart) -> Result<()> {
        // ETA comes from a rolling mean, so the bar's message carries it.
        // The template is a constant. On a template error the default bar
        // still shows progress, so the run continues.
        let style =
            ProgressStyle::with_template("{bar:40.cyan/dim} {pos:>7}/{len:7} [{elapsed}<{msg}")
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("━━─");
        let pb = self.mp.add(ProgressBar::new(start.total_steps as u64));
        pb.set_style(style);
        if start.start_step > 0 {
            pb.set_position(start.start_step as u64);
            self.println(format!("📍 Resuming from step {}", start.start_step));
        }
        self.pb = Some(pb);
        if start.warmup_steps > 0 {
            self.println(format!(
                "🔥 Warmup: {} steps, base LR: {:.6e}",
                start.warmup_steps, start.base_lr
            ));
        }
        Ok(())
    }

    fn on_step(&mut self, report: &StepReport) -> Result<()> {
        self.last = (report.loss, report.grad_norm);
        if let Some(pb) = &self.pb {
            pb.set_position(report.step as u64);
            pb.set_message(step_message(report));
        }
        Ok(())
    }

    fn on_eval(&mut self, step: usize, report: &EvalReport) -> Result<()> {
        let best = report
            .best_loss
            .map_or_else(|| "none".to_string(), |b| format!("{b:.4}"));
        self.println(format!(
            "📏 Eval at step {step}: loss {:.4} | ppl {:.2} | {} tokens | best {best} | {} since best",
            report.loss, report.perplexity, report.tokens, report.since_best
        ));
        Ok(())
    }

    fn on_checkpoint(&mut self, note: &CheckpointNote) -> Result<()> {
        match note {
            CheckpointNote::Saved { step } => {
                self.println(format!("💾 Checkpoint saved: step {step}"));
            }
            CheckpointNote::Best { step } => {
                self.println(format!("🏆 Best checkpoint saved: step {step}"));
            }
            CheckpointNote::Final { path } => {
                if let Some(pb) = &self.pb {
                    pb.finish_with_message(format!(
                        "loss: {:.4} | grad: {:.4} | done ✓",
                        self.last.0, self.last.1
                    ));
                }
                self.println(format!("\n💾 Saving final checkpoint to {path}"));
            }
        }
        Ok(())
    }

    fn on_error(&mut self, step: usize, error: &Error) {
        if let Some(pb) = &self.pb {
            pb.finish_and_clear();
        }
        let err_msg = error.to_string();
        self.println(format!("\n❌ Training error at step {step}:"));
        if err_msg.contains("out of memory") || err_msg.contains("CUDA") || err_msg.contains("cuda")
        {
            self.println(format!("   🔥 CUDA Error: {err_msg}"));
            self.println("   💡 Try: reduce batch_size, seq_len, or use gradient accumulation");
        } else if err_msg.contains("NaN") || err_msg.contains("nan") {
            self.println(format!("   🔥 Numerical Error: {err_msg}"));
            self.println("   💡 Try: lower learning rate or check data");
        } else {
            self.println(format!("   🔥 Error: {err_msg}"));
        }
        self.println("");
    }

    fn on_finish(&mut self, summary: &FitSummary) -> Result<()> {
        if let Some(reason) = &summary.stopped_early {
            self.println(format!(
                "⏹️  Early stop at step {}: {reason}",
                summary.final_step
            ));
        }
        self.println(format!(
            "✅ Training complete | final loss: {:.4} | grad: {:.4}",
            summary.last_loss.unwrap_or(0.0),
            summary.last_grad_norm.unwrap_or(0.0)
        ));
        Ok(())
    }
}

/// The bar message of `report`: ETA, loss, norm, recent throughput, and MFU when configured.
fn step_message(report: &StepReport) -> String {
    let mfu = report
        .mfu
        .map_or_else(String::new, |mfu| format!(" | MFU {:.1}%", mfu * 100.0));
    format!(
        "{}] loss: {:.4} | grad: {:.4} | {:.2} samples/s | {:.0} tok/s{mfu}",
        format_duration(report.eta_secs),
        report.loss,
        report.grad_norm,
        report.recent_samples_per_sec,
        report.recent_tokens_per_sec
    )
}

/// Seconds as the largest two units, such as `5m32s`, `2h14m` or `3d05h`.
fn format_duration(secs: f64) -> String {
    let total_secs = secs.round() as u64;
    let days = total_secs / 86400;
    let hours = (total_secs % 86400) / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;

    if days > 0 {
        format!("{}d{:02}h", days, hours)
    } else if hours > 0 {
        format!("{}h{:02}m", hours, minutes)
    } else if minutes > 0 {
        format!("{}m{:02}s", minutes, seconds)
    } else {
        format!("{}s", seconds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_show_the_two_largest_units() {
        assert_eq!(format_duration(0.0), "0s");
        assert_eq!(format_duration(59.4), "59s");
        assert_eq!(format_duration(332.0), "5m32s");
        assert_eq!(format_duration(2.0 * 3600.0 + 14.0 * 60.0), "2h14m");
        assert_eq!(format_duration(3.0 * 86400.0 + 5.0 * 3600.0), "3d05h");
    }

    #[test]
    fn a_hidden_sink_takes_every_event_without_a_bar_error() {
        let mut sink = TuiSink::hidden();
        let start = FitStart {
            start_step: 2,
            total_steps: 4,
            warmup_steps: 1,
            base_lr: 1e-3,
        };
        sink.on_start(&start).expect("start");
        let report = StepReport {
            step: 3,
            loss: 2.5,
            grad_norm: 0.5,
            learning_rate: 1e-3,
            epoch: 0.5,
            logged: true,
            session_samples_per_sec: 1.0,
            recent_samples_per_sec: 1.0,
            eta_secs: 1.0,
            tokens: 16,
            scored_tokens: None,
            tokens_per_sec: 8.0,
            recent_tokens_per_sec: 8.0,
            mfu: None,
            spike: None,
        };
        assert_eq!(
            step_message(&report),
            "1s] loss: 2.5000 | grad: 0.5000 | 1.00 samples/s | 8 tok/s"
        );
        let with_mfu = StepReport {
            mfu: Some(0.4137),
            ..report.clone()
        };
        assert!(step_message(&with_mfu).ends_with("8 tok/s | MFU 41.4%"));
        sink.on_step(&report).expect("step");
        assert_eq!(sink.last, (2.5, 0.5));
        assert_eq!(sink.pb.as_ref().map(ProgressBar::position), Some(3));
        let path = "ckpt/m_final.safetensors".to_string();
        sink.on_checkpoint(&CheckpointNote::Final { path })
            .expect("final");
        assert!(sink.pb.as_ref().is_some_and(ProgressBar::is_finished));
    }
}
