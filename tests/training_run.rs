//! A headless run through `commands::run_train` records its outcome in `run.json`.
//!
//! The command returns its result after `run.json` is on disk. `main` maps
//! that result to the exit code, so these tests check the code the same way.

mod common;

use common::{data_loader, scratch_dir, tiny_config, tokens, write_config_yaml, write_token_file};
use oxidizr::commands::{run_train, TrainArgs};
use oxidizr::config::LitConfig;
use oxidizr::run_record::{exit_code, RunRecord, RunStatus, StepRecord, STEPS_FILE};
use oxidizr::schedule::LrPlan;
use oxidizr::weight_decay::WeightDecayRule;
use std::path::Path;

/// Headless CPU train args for the files under `dir`.
fn headless_args(dir: &Path, data: &Path, max_steps: usize) -> TrainArgs {
    let cfg = tiny_config(&dir.join("ckpt"), max_steps);
    headless_args_for(dir, data, &cfg, "run")
}

/// Headless CPU train args that run `cfg` and write the run directory `dir/<run>`.
fn headless_args_for(dir: &Path, data: &Path, cfg: &LitConfig, run: &str) -> TrainArgs {
    let cfg_path = dir.join("config.yaml");
    write_config_yaml(&cfg_path, cfg);
    TrainArgs {
        config_file: cfg_path.to_string_lossy().into_owned(),
        data_file: Some(data.to_string_lossy().into_owned()),
        target_device: Some("cpu".to_string()),
        seq_length: None,
        batch_size: None,
        gradient_accumulation: None,
        headless: true,
        max_steps: Some(cfg.trainer.max_steps),
        resume: None,
        gpus: None,
        sync_backend: None,
        prefetch: 0,
        max_checkpoints: None,
        async_checkpoint: false,
        dtype: None,
        checkpoint_dir: None,
        run_dir: Some(dir.join(run).to_string_lossy().into_owned()),
    }
}

/// Every `steps.json` line of the run directory `run_dir`.
fn step_records(run_dir: &Path) -> Vec<StepRecord> {
    std::fs::read_to_string(run_dir.join(STEPS_FILE))
        .expect("steps.json")
        .lines()
        .map(|line| serde_json::from_str(line).expect("parse step line"))
        .collect()
}

/// The plan `LitTrainer` builds for `cfg` over the token stream of `common::data_loader`.
fn lr_plan(cfg: &LitConfig) -> LrPlan {
    let layout = data_loader(cfg).row_layout().expect("row layout");
    let epoch_steps = layout.total_steps(cfg.trainer.gradient_accumulation, cfg.trainer.num_epochs);
    LrPlan::new(&cfg.trainer, epoch_steps).expect("lr plan")
}

/// Six cosine steps after two warmup steps, logged and saved every three steps.
fn logged_cosine_config(dir: &Path) -> LitConfig {
    let mut cfg = tiny_config(&dir.join("ckpt"), 6);
    cfg.trainer.log_interval = 1;
    cfg.trainer.save_interval = 3;
    cfg
}

#[test]
fn steps_json_records_the_learning_rate_each_step_applied() {
    let dir = scratch_dir("headless-run-applied-lr");
    let data = dir.join("tokens.bin");
    let cfg = logged_cosine_config(&dir);
    write_token_file(&data, &tokens(&cfg, 4000));

    let result = run_train(headless_args_for(&dir, &data, &cfg, "run"));

    assert!(result.is_ok(), "training failed: {result:?}");
    let records = step_records(&dir.join("run"));
    assert_eq!(records.len(), 6, "{records:?}");
    let plan = lr_plan(&cfg);
    for record in &records {
        let applied = plan.lr_at(record.step - 1);
        assert_eq!(
            record.learning_rate.to_bits(),
            applied.to_bits(),
            "step {}: logged {:e}, applied {applied:e}",
            record.step,
            record.learning_rate
        );
    }
    // Warmup 2: step 3 is the first post-warmup step and applies the full rate.
    let base = cfg.trainer.learning_rate;
    assert_eq!(records[0].learning_rate, base * 0.5);
    assert_eq!(records[1].learning_rate, base);
    assert_eq!(records[2].learning_rate, base);
    // Steps 4..=6 decay toward min_lr_ratio * base and never reach it.
    for pair in records[2..].windows(2) {
        assert!(pair[1].learning_rate < pair[0].learning_rate, "{pair:?}");
    }
    assert!(records[5].learning_rate > base * cfg.trainer.min_lr_ratio);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_resumed_run_continues_the_schedule_at_the_restored_step() {
    let dir = scratch_dir("headless-run-resumed-lr");
    let data = dir.join("tokens.bin");
    let cfg = logged_cosine_config(&dir);
    write_token_file(&data, &tokens(&cfg, 4000));
    let full = run_train(headless_args_for(&dir, &data, &cfg, "full"));
    assert!(full.is_ok(), "training failed: {full:?}");

    let ckpt = dir.join("ckpt");
    let weights = std::fs::read_dir(&ckpt)
        .expect("checkpoint dir")
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .find(|path| {
            path.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with("_step_3.safetensors"))
        })
        .expect("step-3 weights");
    let mut args = headless_args_for(&dir, &data, &cfg, "resumed");
    args.resume = Some(weights.to_string_lossy().into_owned());
    let resumed = run_train(args);
    assert!(resumed.is_ok(), "resume failed: {resumed:?}");

    let full = step_records(&dir.join("full"));
    let resumed = step_records(&dir.join("resumed"));
    let steps: Vec<usize> = resumed.iter().map(|r| r.step).collect();
    assert_eq!(steps, vec![4, 5, 6]);
    for record in &resumed {
        let reference = &full[record.step - 1];
        assert_eq!(reference.step, record.step);
        assert_eq!(
            record.learning_rate.to_bits(),
            reference.learning_rate.to_bits(),
            "step {}",
            record.step
        );
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_headless_resume_with_another_weight_decay_rule_is_an_error() {
    let dir = scratch_dir("headless-run-decay-mismatch");
    let data = dir.join("tokens.bin");
    let mut cfg = logged_cosine_config(&dir);
    write_token_file(&data, &tokens(&cfg, 4000));
    let full = run_train(headless_args_for(&dir, &data, &cfg, "full"));
    assert!(full.is_ok(), "training failed: {full:?}");
    let meta = std::fs::read_to_string(dir.join("ckpt").join("checkpoint_step_3.meta.json"))
        .expect("step-3 meta");
    assert!(meta.contains(r#""weight_decay_rule": "skip_1d""#), "{meta}");

    cfg.trainer.weight_decay_rule = Some(WeightDecayRule::DecayAll);
    let mut args = headless_args_for(&dir, &data, &cfg, "resumed");
    args.resume = Some("auto".to_string());
    let resumed = run_train(args);
    let err = format!(
        "{:#}",
        resumed.expect_err("a changed rule resumed headless")
    );
    assert!(
        err.contains("decay_all") && err.contains("skip_1d"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_headless_run_writes_a_completed_run_record_and_a_step_log() {
    let dir = scratch_dir("headless-run-completed");
    let data = dir.join("tokens.bin");
    write_token_file(&data, &tokens(&tiny_config(&dir, 2), 4000));

    let result = run_train(headless_args(&dir, &data, 2));

    assert!(result.is_ok(), "training failed: {result:?}");
    assert_eq!(exit_code(&result), 0);
    let run_dir = dir.join("run");
    let record = RunRecord::load(&run_dir).expect("run.json exists after a headless run");
    assert_eq!(record.status, RunStatus::Completed);
    assert_eq!(record.error, None);
    assert_eq!(record.final_step, 2);
    assert!(record.last_loss.is_some_and(f64::is_finite), "{record:?}");
    assert_eq!(record.dataset_size, Some(4000));
    assert!(record.config.is_some(), "{record:?}");
    assert!(!record.started_at.is_empty() && !record.finished_at.is_empty());

    let steps = std::fs::read_to_string(run_dir.join(STEPS_FILE)).expect("headless steps.json");
    let first: StepRecord =
        serde_json::from_str(steps.lines().next().expect("a step line")).expect("parse");
    assert_eq!(first.step, 1);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_data_file_too_small_for_one_batch_records_a_failed_run() {
    let dir = scratch_dir("headless-run-failed");
    let data = dir.join("tokens.bin");
    // Four tokens hold no whole row of `max_seq_len + 1`, so no batch fits.
    write_token_file(&data, &[1, 2, 3, 4]);

    let result = run_train(headless_args(&dir, &data, 2));

    assert!(result.is_err(), "a too-small data file trained");
    assert_ne!(exit_code(&result), 0);
    let record = RunRecord::load(&dir.join("run")).expect("run.json exists after a failed run");
    assert_eq!(record.status, RunStatus::Failed);
    let error = record.error.expect("a failed run records its error");
    assert!(error.contains("whole rows"), "{error}");
    assert_eq!(record.final_step, 0);
    assert_eq!(record.last_loss, None);
    let _ = std::fs::remove_dir_all(dir);
}

/// The `run.json` error of a headless CPU run with `--gpus <gpus>`.
fn gpus_error(tag: &str, gpus: &str) -> String {
    let dir = scratch_dir(tag);
    let data = dir.join("tokens.bin");
    write_token_file(&data, &tokens(&tiny_config(&dir, 2), 4000));
    let mut args = headless_args(&dir, &data, 2);
    args.gpus = Some(gpus.to_string());

    let result = run_train(args);

    assert_ne!(exit_code(&result), 0);
    let record = RunRecord::load(&dir.join("run")).expect("run.json exists after a failed run");
    assert_eq!(record.status, RunStatus::Failed);
    assert_eq!(record.final_step, 0);
    let _ = std::fs::remove_dir_all(dir);
    record.error.expect("a failed run records its error")
}

/// A CPU run cannot take two GPUs. The message names what this build lacks.
#[test]
fn two_gpus_on_a_cpu_run_name_the_missing_support() {
    let error = gpus_error("headless-run-two-gpus", "0,1");
    assert!(error.contains("`--gpus 0,1`"), "{error}");
    assert!(error.contains("--features cuda,nccl"), "{error}");
    assert!(error.contains("--target-device gpu"), "{error}");
    #[cfg(not(feature = "cuda"))]
    assert!(error.contains("no CUDA support"), "{error}");
    #[cfg(feature = "cuda")]
    assert!(error.contains("trains on the CPU"), "{error}");
}

#[test]
fn a_duplicate_gpu_id_is_rejected_on_every_build() {
    let error = gpus_error("headless-run-duplicate-gpus", "0,0");
    assert!(error.contains("`--gpus 0,0`"), "{error}");
    assert!(error.contains("GPU 0 twice"), "{error}");
}

#[test]
fn a_bad_gpu_id_is_rejected() {
    let error = gpus_error("headless-run-bad-gpu", "0,one");
    assert!(error.contains("invalid GPU id 'one'"), "{error}");
}
