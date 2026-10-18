//! Headless output: one JSON line per event on stdout, the error line on stderr.

use super::sink::{CheckpointNote, EvalReport, FitStart, FitSummary, StepReport, StepSink};
use crate::metrics::{device_mib_text, gpu_memory, process_mib_text, GpuMemory, SpikeEvent};
use crate::prelude::{Error, Result};
use std::io::{self, Stderr, Stdout, Write};

/// Writes the headless output of a fit loop to `out` and its error line to `err`.
///
/// Every non-error line goes to `out`. The final-save line is plain text.
pub struct JsonSink<O: Write, E: Write> {
    out: O,
    err: E,
    /// Reads the GPU memory of this process for each logged step. `None` when unknown.
    memory: fn() -> Option<GpuMemory>,
}

impl JsonSink<Stdout, Stderr> {
    /// The sink of `--headless`: stdout, stderr and NVML for memory.
    pub fn stdio() -> Self {
        Self::new(io::stdout(), io::stderr(), gpu_memory)
    }
}

impl<O: Write, E: Write> JsonSink<O, E> {
    pub fn new(out: O, err: E, memory: fn() -> Option<GpuMemory>) -> Self {
        Self { out, err, memory }
    }

    /// The `{"spike": ...}` event line of `event`.
    fn write_spike(&mut self, event: &SpikeEvent) -> Result<()> {
        writeln!(
            self.out,
            "{{\"spike\": {{\"step\": {}, \"loss\": {}, \"mean\": {}, \"std\": {}, \"z\": {}, \
             \"action\": \"{}\"}}}}",
            event.step,
            fixed4(event.loss),
            fixed4(event.mean),
            fixed4(event.std),
            fixed4(event.z()),
            event.action.name()
        )?;
        Ok(())
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
        if let Some(event) = &report.spike {
            self.write_spike(event)?;
        }
        if !report.logged {
            return Ok(());
        }
        let memory = (self.memory)();
        // New keys go after `vram`, so the keys before it keep their order.
        let mut tail = String::new();
        if let Some(scored) = report.scored_tokens {
            tail.push_str(&format!(", \"scored_tokens\": {scored}"));
        }
        if let Some(mfu) = report.mfu {
            tail.push_str(&format!(", \"mfu\": {}", fixed4(mfu)));
        }
        writeln!(
            self.out,
            "{{\"step\": {}, \"loss\": {:.4}, \"grad_norm\": {:.4}, \"learning_rate\": {:.6e}, \
             \"epoch\": {:.2}, \"samples/s\": {:.2}, \"vram\": \"{}\", \"tokens\": {}, \
             \"tokens_per_sec\": {:.2}, \"vram_device\": \"{}\"{tail}}}",
            report.step,
            report.loss,
            report.grad_norm,
            report.learning_rate,
            report.epoch,
            report.session_samples_per_sec,
            process_mib_text(memory),
            report.tokens,
            report.tokens_per_sec,
            device_mib_text(memory),
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
            tokens: 2048,
            scored_tokens: None,
            tokens_per_sec: 395.6736,
            recent_tokens_per_sec: 1152.0,
            mfu: None,
            spike: None,
        }
    }

    fn no_gpu() -> Option<GpuMemory> {
        None
    }

    fn one_gpu() -> Option<GpuMemory> {
        const MIB: u64 = 1024 * 1024;
        Some(GpuMemory {
            process_bytes: Some(812 * MIB),
            devices: 1,
            device_used_bytes: 1024 * MIB,
            device_total_bytes: 12288 * MIB,
        })
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
        let mut sink = JsonSink::new(&mut out, &mut err, no_gpu);
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
                r#"{"step": 21, "loss": 11.4948, "grad_norm": 0.1980, "learning_rate": 2.000000e-3, "epoch": 0.21, "samples/s": 3.09, "vram": "n/a", "tokens": 2048, "tokens_per_sec": 395.67, "vram_device": "n/a"}"#,
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
        let mut sink = JsonSink::new(&mut out, &mut err, no_gpu);
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
        let mut sink = JsonSink::new(&mut out, &mut err, no_gpu);
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
        let mut sink = JsonSink::new(&mut out, &mut err, no_gpu);
        let start = FitStart {
            start_step: 3,
            total_steps: 6,
            warmup_steps: 0,
            base_lr: 1e-3,
        };
        sink.on_start(&start).expect("start");
        assert!(out.is_empty());
    }

    #[test]
    fn a_masked_step_with_mfu_appends_its_keys_after_vram_device() {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let mut sink = JsonSink::new(&mut out, &mut err, one_gpu);
        let step = StepReport {
            scored_tokens: Some(731),
            mfu: Some(0.41237),
            ..report(true)
        };
        sink.on_step(&step).expect("step");
        assert_eq!(
            lines(&out),
            [
                r#"{"step": 21, "loss": 11.4948, "grad_norm": 0.1980, "learning_rate": 2.000000e-3, "epoch": 0.21, "samples/s": 3.09, "vram": "812", "tokens": 2048, "tokens_per_sec": 395.67, "vram_device": "1024/12288", "scored_tokens": 731, "mfu": 0.4124}"#
            ]
        );
        let parsed: serde_json::Value = serde_json::from_str(&lines(&out)[0]).expect("JSON");
        assert_eq!(parsed["scored_tokens"], 731);
        assert!(err.is_empty());
    }

    #[test]
    fn a_spike_writes_its_event_line_on_every_step() {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let mut sink = JsonSink::new(&mut out, &mut err, no_gpu);
        let spike = SpikeEvent {
            step: 21,
            loss: 11.49481,
            mean: 3.5,
            std: 0.0,
            action: crate::metrics::SpikeAction::Skip,
        };
        let step = StepReport {
            spike: Some(spike),
            ..report(false)
        };
        sink.on_step(&step).expect("silent step");
        assert_eq!(
            lines(&out),
            [
                r#"{"spike": {"step": 21, "loss": 11.4948, "mean": 3.5000, "std": 0.0000, "z": null, "action": "skip"}}"#
            ]
        );
        assert!(err.is_empty());
    }
}
