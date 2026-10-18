//! One fit loop drives both output modes.
//!
//! The headless sink writes the documented stdout lines. Both sinks see the
//! same steps, so `steps.json` is identical in both modes, on a resume too.

mod common;

use common::{
    build_model, data_loader, scratch_dir, steps_without_timing, tiny_config, train_parts, trainer,
};
use oxidizr::config::LitConfig;
use oxidizr::fit::{JsonSink, StepSink, TuiSink};
use oxidizr::metrics::GpuMemory;
use oxidizr::run_record::{StepRecord, STEPS_FILE};
use std::path::Path;

fn no_gpu() -> Option<GpuMemory> {
    None
}

/// Trains `cfg` from `start_step` through `sink` and returns the `steps.json` text.
///
/// `steps.json` lands in the checkpoint directory of `cfg`.
fn fit(cfg: &LitConfig, start_step: usize, sink: &mut dyn StepSink) -> String {
    let model = build_model(cfg);
    let mut parts = train_parts(&model);
    let mut loader = data_loader(cfg);
    let layout = loader.row_layout().expect("row layout");
    let position = layout.position_after_steps(start_step, cfg.trainer.gradient_accumulation);
    loader.seek(position).expect("seek");
    {
        let mut t = trainer(cfg, &mut parts, layout, start_step, position);
        t.fit_with_sink(&model, &mut loader, sink).expect("fit");
        assert_eq!(t.global_step(), cfg.trainer.max_steps);
    }
    let steps = Path::new(&cfg.trainer.checkpoint_dir).join(STEPS_FILE);
    std::fs::read_to_string(steps).expect("steps.json")
}

fn records(steps_json: &str) -> Vec<StepRecord> {
    steps_json
        .lines()
        .map(|line| serde_json::from_str(line).expect("parse step line"))
        .collect()
}

fn config(checkpoint_dir: &Path, max_steps: usize, log_interval: usize) -> LitConfig {
    let mut cfg = tiny_config(checkpoint_dir, max_steps);
    cfg.trainer.log_interval = log_interval;
    cfg
}

/// The headless step line of `record`, with the run's own `samples/s` value.
///
/// `tokens_per_sec` comes from `record`, the `steps.json` line of the same step.
fn step_line(record: &StepRecord, samples_per_sec: f64) -> String {
    format!(
        "{{\"step\": {}, \"loss\": {:.4}, \"grad_norm\": {:.4}, \"learning_rate\": {:.6e}, \
         \"epoch\": {:.2}, \"samples/s\": {:.2}, \"vram\": \"n/a\", \"tokens\": {}, \
         \"tokens_per_sec\": {:.2}, \"vram_device\": \"n/a\"}}",
        record.step,
        record.loss,
        record.grad_norm,
        record.learning_rate,
        record.epoch,
        samples_per_sec,
        record.tokens,
        record.tokens_per_sec
    )
}

#[test]
fn headless_stdout_has_the_documented_lines_in_order() {
    let dir = scratch_dir("fit-headless-lines");
    let mut cfg = config(&dir.join("ckpt"), 3, 1);
    cfg.trainer.save_interval = 2;
    let (mut out, mut err) = (Vec::new(), Vec::new());

    let steps_json = fit(&cfg, 0, &mut JsonSink::new(&mut out, &mut err, no_gpu));

    let stdout = String::from_utf8(out).expect("utf-8 stdout");
    let lines: Vec<&str> = stdout.lines().collect();
    let records = records(&steps_json);
    assert_eq!(records.len(), 3, "{steps_json}");
    assert_eq!(lines.len(), 7, "{stdout}");
    assert_eq!(
        lines[0],
        r#"{"warmup": {"steps": 2, "base_lr": 3.000000e-3}}"#
    );
    // 2 rows x 2 micro-batches x 8 tokens of one rank.
    let tokens = cfg.trainer.batch_size * cfg.trainer.gradient_accumulation * cfg.model.max_seq_len;
    for (line, record) in [lines[1], lines[2], lines[4]].into_iter().zip(&records) {
        let parsed: serde_json::Value = serde_json::from_str(line).expect("a JSON step line");
        let samples = parsed["samples/s"].as_f64().expect("samples/s");
        assert_eq!(line, step_line(record, samples));
        assert_eq!(record.tokens, tokens);
        assert!(record.tokens_per_sec > 0.0, "{record:?}");
        assert_eq!(
            record.scored_tokens, None,
            "an unmasked run scores every token"
        );
        assert_eq!(record.mfu, None, "no peak_tflops, no MFU");
    }
    assert_eq!(lines[3], r#"{"checkpoint": {"step": 2}}"#);
    let prefix = cfg.model.checkpoint_prefix();
    let final_path = format!("{}/{prefix}_final.safetensors", cfg.trainer.checkpoint_dir);
    assert_eq!(lines[5], format!("Saving final checkpoint to {final_path}"));
    assert_eq!(lines[6], r#"{"status": "complete"}"#);
    assert!(Path::new(&final_path).exists(), "{final_path}");
    assert!(err.is_empty(), "{}", String::from_utf8_lossy(&err));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn both_output_modes_write_the_same_steps_json() {
    let dir = scratch_dir("fit-modes-steps-json");
    let headless_cfg = config(&dir.join("headless"), 5, 2);
    let interactive_cfg = config(&dir.join("interactive"), 5, 2);
    let (mut out, mut err) = (Vec::new(), Vec::new());

    let headless = fit(
        &headless_cfg,
        0,
        &mut JsonSink::new(&mut out, &mut err, no_gpu),
    );
    let interactive = fit(&interactive_cfg, 0, &mut TuiSink::hidden());

    assert_eq!(
        steps_without_timing(&headless),
        steps_without_timing(&interactive)
    );
    let steps: Vec<usize> = records(&headless).iter().map(|r| r.step).collect();
    assert_eq!(steps, vec![1, 3, 5]);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_resumed_session_logs_its_first_step_in_both_modes() {
    let dir = scratch_dir("fit-resume-first-step");
    let headless_cfg = config(&dir.join("headless"), 6, 10);
    let interactive_cfg = config(&dir.join("interactive"), 6, 10);
    let (mut out, mut err) = (Vec::new(), Vec::new());

    let headless = fit(
        &headless_cfg,
        3,
        &mut JsonSink::new(&mut out, &mut err, no_gpu),
    );
    let interactive = fit(&interactive_cfg, 3, &mut TuiSink::hidden());

    assert_eq!(
        steps_without_timing(&headless),
        steps_without_timing(&interactive)
    );
    let steps: Vec<usize> = records(&headless).iter().map(|r| r.step).collect();
    assert_eq!(steps, vec![4]);
    let stdout = String::from_utf8(out).expect("utf-8 stdout");
    let first = stdout.lines().nth(1).expect("a step line");
    assert!(first.starts_with(r#"{"step": 4, "#), "{stdout}");
    let _ = std::fs::remove_dir_all(dir);
}
