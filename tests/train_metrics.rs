//! Token counts, tokens per second, MFU and the guard counters in `steps.json`, `run.json` and the meta.

mod common;

use boostr::CpuRuntime;
use common::{build_model, scratch_dir, tiny_config, tokens, write_config_yaml, write_token_file};
use oxidizr::commands::{run_train, run_train_with_group, TrainArgs};
use oxidizr::config::{CheckpointMeta, LitConfig};
use oxidizr::dp::RankGroup;
use oxidizr::metrics::{model_flops_per_token, MetricsConfig, SpikeGuardConfig};
use oxidizr::run_record::{RunRecord, StepRecord, STEPS_FILE};
use std::path::Path;
use std::time::Duration;

/// Headless CPU train args that run `cfg` and write the run directory `dir/run`.
fn args(dir: &Path, cfg: &LitConfig) -> TrainArgs {
    let data = dir.join("tokens.bin");
    write_token_file(&data, &tokens(cfg, 4000));
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
        run_dir: Some(dir.join("run").to_string_lossy().into_owned()),
    }
}

fn step_records(run_dir: &Path) -> Vec<StepRecord> {
    std::fs::read_to_string(run_dir.join(STEPS_FILE))
        .expect("steps.json")
        .lines()
        .map(|line| serde_json::from_str(line).expect("parse step line"))
        .collect()
}

/// `tiny_config` logged every step, saved at step 2, with a peak and a guard.
fn metrics_config(dir: &Path) -> LitConfig {
    let mut cfg = tiny_config(&dir.join("ckpt"), 3);
    cfg.trainer.log_interval = 1;
    cfg.trainer.save_interval = 2;
    cfg.trainer.metrics = Some(MetricsConfig {
        peak_tflops: Some(1.0),
    });
    cfg.trainer.spike_guard = Some(SpikeGuardConfig::default());
    cfg
}

/// Hand count for `tiny_model_config(false)`:
/// per layer q, k, v, o at 16 x 16 (1024) plus gate, up, down at 32 x 16 (1536).
/// Two layers give 5120, the untied head 24 x 16 adds 384: N = 5504, 6N = 33024.
/// Attention: 12 * 2 layers * width 16 * span 8 = 3072. Total 36096.
#[test]
fn the_tiny_model_costs_its_hand_counted_flops_per_token() {
    let dir = scratch_dir("metrics-flops");
    let cfg = tiny_config(&dir, 1);
    let model = build_model(&cfg);
    let flops = model_flops_per_token(&model, &cfg.model).expect("flops");
    assert_eq!(flops, 36096.0);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_run_records_tokens_mfu_and_guard_counters() {
    let dir = scratch_dir("metrics-run");
    let cfg = metrics_config(&dir);
    run_train(args(&dir, &cfg)).expect("training run");

    let run_dir = dir.join("run");
    // 2 rows x 2 micro-batches x 8 tokens, one rank.
    let tokens = 2 * 2 * 8;
    let records = step_records(&run_dir);
    assert_eq!(records.len(), 3);
    for record in &records {
        assert_eq!(record.tokens, tokens, "{record:?}");
        assert!(record.tokens_per_sec > 0.0, "{record:?}");
        let mfu = record.mfu.expect("peak_tflops is set");
        let want = 36096.0 * record.tokens_per_sec / 1e12;
        assert!(
            (mfu - want).abs() <= 1e-12 * want.max(1.0),
            "{mfu} != {want}"
        );
    }

    let run = RunRecord::load(&run_dir).expect("run.json");
    assert!(run.tokens_per_sec.is_some_and(|r| r > 0.0), "{run:?}");
    assert!(run.mfu.is_some_and(|m| m > 0.0), "{run:?}");
    // min_steps 50: three steps flag nothing.
    assert_eq!(
        (run.spikes_detected, run.spikes_skipped),
        (Some(0), Some(0))
    );

    let meta_path = Path::new(&cfg.trainer.checkpoint_dir).join("checkpoint_step_2.meta.json");
    let meta = CheckpointMeta::load(&meta_path.to_string_lossy()).expect("meta");
    let state = meta.spike_guard.expect("the meta records the guard");
    assert_eq!((state.observed, state.detected, state.skipped), (2, 0, 0));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_run_without_metrics_keys_records_no_mfu_and_no_counters() {
    let dir = scratch_dir("metrics-off");
    let mut cfg = tiny_config(&dir.join("ckpt"), 2);
    cfg.trainer.log_interval = 1;
    run_train(args(&dir, &cfg)).expect("training run");

    let run_dir = dir.join("run");
    let text = std::fs::read_to_string(run_dir.join("run.json")).expect("run.json");
    for key in ["\"mfu\"", "spikes_detected", "spikes_skipped"] {
        assert!(!text.contains(key), "{key} in {text}");
    }
    assert!(text.contains("tokens_per_sec"), "{text}");
    let steps = std::fs::read_to_string(run_dir.join(STEPS_FILE)).expect("steps.json");
    assert!(
        !steps.contains("mfu") && !steps.contains("scored_tokens"),
        "{steps}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// The token count covers every rank: 2 rows x 2 micro-batches x 2 ranks x 8 tokens.
#[test]
fn a_world_two_step_counts_the_tokens_of_every_rank() {
    let dir = scratch_dir("metrics-world-two");
    let mut cfg = tiny_config(&dir.join("ckpt"), 2);
    cfg.trainer.log_interval = 1;
    let group = RankGroup::local(2, Duration::from_secs(120)).expect("local group");
    run_train_with_group::<CpuRuntime>(args(&dir, &cfg), group).expect("world-2 run");

    let records = step_records(&dir.join("run"));
    assert_eq!(records.len(), 2);
    for record in records {
        assert_eq!(record.tokens, 2 * 2 * 2 * 8, "{record:?}");
    }
    let _ = std::fs::remove_dir_all(dir);
}
