//! Headless output: one JSON line per event on stdout, the error line on stderr.

use super::sink::{CheckpointNote, EvalReport, FitStart, FitSummary, StepReport, StepSink};
use crate::prelude::{Error, Result};
use std::io::{self, Stderr, Stdout, Write};

/// Writes the headless output of a fit loop to `out` and its error line to `err`.
///
/// Every non-error line goes to `out`. The final-save line is plain text.
pub struct JsonSink<O: Write, E: Write> {
    out: O,
    err: E,
    /// Reports device memory in use, in MiB, for each logged step.
    vram: fn() -> String,
}

impl JsonSink<Stdout, Stderr> {
    /// The sink of `--headless`: stdout, stderr and `nvidia-smi` for memory.
    pub fn stdio() -> Self {
        Self::new(io::stdout(), io::stderr(), query_vram)
    }
}

impl<O: Write, E: Write> JsonSink<O, E> {
    pub fn new(out: O, err: E, vram: fn() -> String) -> Self {
        Self { out, err, vram }
    }
}

impl<O: Write, E: Write> StepSink for JsonSink<O, E> {
    fn on_start(&mut self, start: &FitStart) -> Result<()> {
        if start.warmup_steps > 0 {
            writeln!(
                self.out,
                "{{\"warmup\": {{\"steps\": {}, \"base_lr\": {:.6e}}}}}",
                start.warmup_steps, start.base_lr
            )?;
        }
        Ok(())
    }

    fn on_step(&mut self, report: &StepReport) -> Result<()> {
        if !report.logged {
            return Ok(());
        }
        let vram = (self.vram)();
        writeln!(
            self.out,
            "{{\"step\": {}, \"loss\": {:.4}, \"grad_norm\": {:.4}, \"learning_rate\": {:.6e}, \
             \"epoch\": {:.2}, \"samples/s\": {:.2}, \"vram\": \"{}\"}}",
            report.step,
            report.loss,
            report.grad_norm,
            report.learning_rate,
            report.epoch,
            report.session_samples_per_sec,
            vram
        )?;
        Ok(())
    }

    fn on_eval(&mut self, step: usize, report: &EvalReport) -> Result<()> {
        let best = report.best_loss.map_or_else(|| "null".to_string(), fixed4);
        writeln!(
            self.out,
            "{{\"eval\": {{\"step\": {step}, \"loss\": {}, \"perplexity\": {}, \"batches\": {}, \
             \"tokens\": {}, \"best_loss\": {best}, \"since_best\": {}}}}}",
            fixed4(report.loss),
            fixed4(report.perplexity),
            report.batches,
            report.tokens,
            report.since_best
        )?;
        Ok(())
    }

    fn on_checkpoint(&mut self, note: &CheckpointNote) -> Result<()> {
        match note {
            CheckpointNote::Saved { step } => {
                writeln!(self.out, "{{\"checkpoint\": {{\"step\": {step}}}}}")?
            }
            CheckpointNote::Best { step } => writeln!(
                self.out,
                "{{\"checkpoint\": {{\"step\": {step}, \"label\": \"best\"}}}}"
            )?,
            CheckpointNote::Final { path } => {
                writeln!(self.out, "Saving final checkpoint to {path}")?
            }
        }
        Ok(())
    }

    fn on_error(&mut self, step: usize, error: &Error) {
        let line = serde_json::json!({
            "error": format!("Training error at step {step}"),
            "message": error.to_string(),
        });
        // The step error is what the run returns. A lost stderr line cannot replace it.
        let _ = writeln!(self.err, "{line}");
    }

    fn on_finish(&mut self, summary: &FitSummary) -> Result<()> {
        match &summary.stopped_early {
            None => writeln!(self.out, "{{\"status\": \"complete\"}}")?,
            Some(reason) => writeln!(
                self.out,
                "{{\"status\": \"complete\", \"stopped_early\": true, \"reason\": {}}}",
                serde_json::Value::from(reason.as_str())
            )?,
        }
        Ok(())
    }
}

/// `value` with four decimals, or `null` when it is not finite, so the line stays JSON.
fn fixed4(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.4}")
    } else {
        "null".to_string()
    }
}

/// Memory of the first compute process `nvidia-smi` lists, in MiB. `"0"` when unknown.
fn query_vram() -> String {
    std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.lines().next().map(|l| l.trim().to_string()))
        .unwrap_or_else(|| "0".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(logged: bool) -> StepReport {
        StepReport {
            step: 21,
            loss: 11.49481,
            grad_norm: 0.19804,
            learning_rate: 2e-3,
            epoch: 0.2149,
            logged,
            session_samples_per_sec: 3.0912,
            recent_samples_per_sec: 9.0,
            eta_secs: 1.0,
        }
    }

    fn zero_vram() -> String {
        "0".to_string()
    }

    fn lines(bytes: &[u8]) -> Vec<String> {
        String::from_utf8(bytes.to_vec())
            .expect("utf-8 output")
            .lines()
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn every_event_writes_the_documented_line() {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let mut sink = JsonSink::new(&mut out, &mut err, zero_vram);
        let start = FitStart {
            start_step: 0,
            total_steps: 30,
            warmup_steps: 2,
            base_lr: 3e-3,
        };
        sink.on_start(&start).expect("start");
        sink.on_step(&report(true)).expect("logged step");
        sink.on_step(&report(false)).expect("silent step");
        sink.on_checkpoint(&CheckpointNote::Saved { step: 21 })
            .expect("checkpoint");
        let path = "ckpt/m_final.safetensors".to_string();
        sink.on_checkpoint(&CheckpointNote::Final { path })
            .expect("final");
        let summary = FitSummary {
            final_step: 30,
            last_loss: Some(1.0),
            last_grad_norm: Some(1.0),
            stopped_early: None,
        };
        sink.on_finish(&summary).expect("finish");

        assert_eq!(
            lines(&out),
            [
                r#"{"warmup": {"steps": 2, "base_lr": 3.000000e-3}}"#,
                r#"{"step": 21, "loss": 11.4948, "grad_norm": 0.1980, "learning_rate": 2.000000e-3, "epoch": 0.21, "samples/s": 3.09, "vram": "0"}"#,
                r#"{"checkpoint": {"step": 21}}"#,
                "Saving final checkpoint to ckpt/m_final.safetensors",
                r#"{"status": "complete"}"#,
            ]
        );
        assert!(err.is_empty());
    }

    #[test]
    fn a_step_error_writes_one_json_line_to_stderr() {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let mut sink = JsonSink::new(&mut out, &mut err, zero_vram);
        let error = Error::TrainingError {
            reason: "NaN loss detected".to_string(),
        };
        sink.on_error(4, &error);

        assert!(out.is_empty());
        assert_eq!(
            lines(&err),
            [
                r#"{"error":"Training error at step 4","message":"training error: NaN loss detected"}"#
            ]
        );
    }

    #[test]
    fn eval_best_and_early_stop_events_write_json_lines() {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let mut sink = JsonSink::new(&mut out, &mut err, zero_vram);
        let mut report = EvalReport {
            loss: 2.3125,
            perplexity: 10.0625,
            batches: 50,
            tokens: 51200,
            best_loss: Some(2.3125),
            since_best: 0,
        };
        sink.on_eval(500, &report).expect("eval");
        sink.on_checkpoint(&CheckpointNote::Best { step: 500 })
            .expect("best");
        report.loss = f64::INFINITY;
        report.perplexity = f64::INFINITY;
        report.since_best = 1;
        sink.on_eval(1000, &report).expect("non-finite eval");
        let summary = FitSummary {
            final_step: 1000,
            last_loss: Some(1.0),
            last_grad_norm: Some(1.0),
            stopped_early: Some("no \"gain\" in 5 evals".to_string()),
        };
        sink.on_finish(&summary).expect("finish");

        let lines = lines(&out);
        assert_eq!(
            lines,
            [
                r#"{"eval": {"step": 500, "loss": 2.3125, "perplexity": 10.0625, "batches": 50, "tokens": 51200, "best_loss": 2.3125, "since_best": 0}}"#,
                r#"{"checkpoint": {"step": 500, "label": "best"}}"#,
                r#"{"eval": {"step": 1000, "loss": null, "perplexity": null, "batches": 50, "tokens": 51200, "best_loss": 2.3125, "since_best": 1}}"#,
                r#"{"status": "complete", "stopped_early": true, "reason": "no \"gain\" in 5 evals"}"#,
            ]
        );
        for line in &lines {
            serde_json::from_str::<serde_json::Value>(line).expect("every line is JSON");
        }
        assert!(err.is_empty());
    }

    #[test]
    fn no_warmup_writes_no_warmup_line() {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let mut sink = JsonSink::new(&mut out, &mut err, zero_vram);
        let start = FitStart {
            start_step: 3,
            total_steps: 6,
            warmup_steps: 0,
            base_lr: 1e-3,
        };
        sink.on_start(&start).expect("start");
        assert!(out.is_empty());
    }
}
