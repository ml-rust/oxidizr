//! A headless run through `commands::run_train` records its outcome in `run.json`.
//!
//! The command returns its result after `run.json` is on disk. `main` maps
//! that result to the exit code, so these tests check the code the same way.

mod common;

use common::{scratch_dir, tiny_config, tokens, write_config_yaml, write_token_file};
use oxidizr::commands::{run_train, TrainArgs};
use oxidizr::run_record::{exit_code, RunRecord, RunStatus, StepRecord, STEPS_FILE};
use std::path::Path;

/// Headless CPU train args for the files under `dir`.
fn headless_args(dir: &Path, data: &Path, max_steps: usize) -> TrainArgs {
    let cfg_path = dir.join("config.yaml");
    write_config_yaml(&cfg_path, &tiny_config(&dir.join("ckpt"), max_steps));
    TrainArgs {
        config_file: cfg_path.to_string_lossy().into_owned(),
        data_file: Some(data.to_string_lossy().into_owned()),
        target_device: Some("cpu".to_string()),
        seq_length: None,
        batch_size: None,
        gradient_accumulation: None,
        headless: true,
        max_steps: Some(max_steps),
        resume: None,
        gpus: None,
        sync_backend: None,
        prefetch: 0,
        max_checkpoints: None,
        dtype: None,
        checkpoint_dir: None,
        run_dir: Some(dir.join("run").to_string_lossy().into_owned()),
    }
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

#[test]
fn two_gpus_are_rejected_with_a_single_gpu_instruction() {
    let dir = scratch_dir("headless-run-two-gpus");
    let data = dir.join("tokens.bin");
    write_token_file(&data, &tokens(&tiny_config(&dir, 2), 4000));
    let mut args = headless_args(&dir, &data, 2);
    args.gpus = Some("0,1".to_string());

    let result = run_train(args);

    assert_ne!(exit_code(&result), 0);
    let record = RunRecord::load(&dir.join("run")).expect("run.json exists after a failed run");
    assert_eq!(record.status, RunStatus::Failed);
    let error = record.error.expect("a failed run records its error");
    assert!(error.contains("`--gpus 0,1`"), "{error}");
    assert!(error.contains("not implemented in this build"), "{error}");
    assert!(error.contains("single GPU"), "{error}");
    assert_eq!(record.final_step, 0);
    let _ = std::fs::remove_dir_all(dir);
}
