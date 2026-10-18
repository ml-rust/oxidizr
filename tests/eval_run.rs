//! Validation evals inside the fit loop: determinism, isolation from training,
//! the held-out split, early stopping, and the best checkpoint.

mod common;

use boostr::{CpuDevice, CpuRuntime};
use common::{
    bits_by_name, build_model, data_loader, eval_loaders, holdout_eval, scratch_dir,
    steps_without_timing, tiny_config, train_parts, trainer, write_config_yaml,
};
use oxidizr::checkpoint_set::{remove_old_checkpoints, CheckpointLabel};
use oxidizr::commands::train_data::{run_loaders, TokenSource};
use oxidizr::config::{find_latest_checkpoint, CheckpointMeta, LitConfig};
use oxidizr::data::{DataPosition, LitDataLoader, RowOrder};
use oxidizr::eval::{EarlyStoppingConfig, Evaluator};
use oxidizr::fit::{JsonSink, StepSink, TuiSink};
use oxidizr::metrics::GpuMemory;
use oxidizr::run_record::{EvalRecord, EVALS_FILE, STEPS_FILE};
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::Path;

const STEPS: usize = 6;
const HOLDOUT: usize = 40;

fn no_gpu() -> Option<GpuMemory> {
    None
}

/// `tiny_config` with a held-out eval after every step.
fn eval_config(dir: &Path) -> LitConfig {
    let mut cfg = tiny_config(dir, STEPS);
    cfg.trainer.log_interval = 1;
    cfg.trainer.eval = Some(holdout_eval(HOLDOUT, 1, 2));
    cfg
}

/// What one fit run left behind.
struct Fitted {
    bits: BTreeMap<String, Vec<u32>>,
    final_step: usize,
    stop_reason: Option<String>,
}

/// Fits `cfg` from step 0 over `train`, with `evaluator` attached when given.
fn fit(
    cfg: &LitConfig,
    train: &mut LitDataLoader<CpuRuntime>,
    evaluator: Option<Evaluator<CpuRuntime>>,
    sink: &mut dyn StepSink,
) -> Fitted {
    let model = build_model(cfg);
    let mut parts = train_parts(&model);
    let layout = train.row_layout().expect("row layout");
    let (final_step, stop_reason) = {
        let mut t = trainer(cfg, &mut parts, layout, 0, DataPosition::default());
        if let Some(evaluator) = evaluator {
            t.attach_eval(evaluator);
        }
        t.fit_with_sink(&model, train, sink).expect("fit");
        (t.global_step(), t.early_stop_reason().map(str::to_string))
    };
    Fitted {
        bits: bits_by_name(&parts),
        final_step,
        stop_reason,
    }
}

fn read(dir: &Path, file: &str) -> String {
    fs::read_to_string(dir.join(file)).unwrap_or_default()
}

fn evals(dir: &Path) -> Vec<EvalRecord> {
    read(dir, EVALS_FILE)
        .lines()
        .map(|l| serde_json::from_str(l).expect("an evals.json line"))
        .collect()
}

#[test]
fn two_evals_on_the_same_weights_are_bit_identical() {
    let dir = scratch_dir("eval-deterministic");
    let cfg = eval_config(&dir);
    let (train, evaluator) = eval_loaders(&cfg);
    let model = build_model(&cfg);
    let mut parts = train_parts(&model);
    let before = bits_by_name(&parts);
    {
        let layout = train.row_layout().expect("row layout");
        let mut t = trainer(&cfg, &mut parts, layout, 0, DataPosition::default());
        t.attach_eval(evaluator);
        let first = t.evaluate_now(&model).expect("eval").expect("attached");
        let second = t.evaluate_now(&model).expect("eval").expect("attached");
        assert_eq!(first.loss_sum.to_bits(), second.loss_sum.to_bits());
        assert_eq!(first.tokens, second.tokens);
        assert_eq!(first.batches, 2);
        assert_eq!(
            first.tokens,
            2 * cfg.trainer.batch_size * cfg.model.max_seq_len
        );
        assert!(first.loss().is_finite() && first.loss() > 0.0);
        assert_eq!(first.perplexity().to_bits(), first.loss().exp().to_bits());
    }
    assert_eq!(before, bits_by_name(&parts), "an eval changed the weights");
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn evals_leave_the_training_run_bit_identical() {
    let dir = scratch_dir("eval-isolated");
    let with_dir = dir.join("with");
    let without_dir = dir.join("without");
    let with_cfg = eval_config(&with_dir);
    let mut without_cfg = eval_config(&without_dir);

    let (mut train, evaluator) = eval_loaders(&with_cfg);
    let with = fit(
        &with_cfg,
        &mut train,
        Some(evaluator),
        &mut TuiSink::hidden(),
    );
    // The same training rows, no eval attached.
    let (mut train, _) = eval_loaders(&without_cfg);
    without_cfg.trainer.eval = None;
    let without = fit(&without_cfg, &mut train, None, &mut TuiSink::hidden());

    assert_eq!(with.bits, without.bits);
    assert_eq!(
        steps_without_timing(&read(&with_dir, STEPS_FILE)),
        steps_without_timing(&read(&without_dir, STEPS_FILE))
    );
    assert_eq!(evals(&with_dir).len(), STEPS);
    assert!(!without_dir.join(EVALS_FILE).exists());
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn held_out_rows_never_reach_a_training_batch() {
    let dir = scratch_dir("eval-disjoint");
    let cfg = eval_config(&dir);
    // Every token is its own stream offset, so a row's first token names the row.
    let source = TokenSource::Generated((0..4000).collect());
    let loaders = run_loaders::<CpuRuntime>(&cfg, &source, &CpuDevice::new()).expect("run loaders");
    let mut train = loaders.train.with_row_order(RowOrder::Shuffled { seed: 7 });
    let evaluator = loaders.evaluator.expect("an eval");
    let total = data_loader(&cfg).total_rows();
    let train_rows = total - HOLDOUT;
    assert_eq!(loaders.holdout_rows, HOLDOUT);
    assert_eq!(train.row_range(), (0, train_rows));
    assert_eq!(evaluator.row_range(), (train_rows, total));
    assert_eq!(evaluator.holdout_rows(), HOLDOUT);

    let width = cfg.model.max_seq_len + 1;
    let per_epoch = train.row_layout().expect("layout").batches_per_epoch();
    let mut seen = HashSet::new();
    for _ in 0..2 * per_epoch {
        let batch = train.next_batch().expect("batch");
        let ids: Vec<i64> = batch.inputs.to_vec();
        for row in ids.chunks(cfg.model.max_seq_len) {
            let index = row[0] as usize / width;
            assert!(index < train_rows, "held-out row {index} reached training");
            seen.insert(index);
        }
    }
    assert_eq!(seen.len(), per_epoch * cfg.trainer.batch_size);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn early_stopping_ends_the_run_cleanly_at_the_expected_eval() {
    let dir = scratch_dir("eval-early-stop");
    let mut cfg = eval_config(&dir);
    // No eval after the first can beat it by 1e9: evals 2 and 3 spend a patience of 2.
    if let Some(eval) = cfg.trainer.eval.as_mut() {
        eval.early_stopping = Some(EarlyStoppingConfig {
            patience: 2,
            min_delta: 1e9,
        });
    }
    let (mut train, evaluator) = eval_loaders(&cfg);
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let run = fit(
        &cfg,
        &mut train,
        Some(evaluator),
        &mut JsonSink::new(&mut out, &mut err, no_gpu),
    );

    assert_eq!(run.final_step, 3);
    assert!(run.stop_reason.is_some());
    let since: Vec<usize> = evals(&dir).iter().map(|e| e.since_best).collect();
    assert_eq!(since, [0, 1, 2]);
    let stdout = String::from_utf8(out).expect("utf-8 stdout");
    let last = stdout.lines().last().expect("a status line");
    let status: serde_json::Value = serde_json::from_str(last).expect("JSON status line");
    assert_eq!(status["status"], "complete");
    assert_eq!(status["stopped_early"], true);
    assert!(status["reason"]
        .as_str()
        .is_some_and(|r| r.contains("2 evals")));
    let eval_lines = stdout
        .lines()
        .filter(|l| l.starts_with(r#"{"eval": "#))
        .count();
    assert_eq!(eval_lines, 3, "{stdout}");

    let final_meta = dir.join(CheckpointLabel::Final.meta_file_name());
    let meta = CheckpointMeta::load(&final_meta.to_string_lossy()).expect("final meta");
    assert_eq!(meta.step, 3);
    assert_eq!(meta.evals_since_best, Some(2));
    assert_eq!(meta.holdout_rows, Some(HOLDOUT));
    assert!(err.is_empty(), "{}", String::from_utf8_lossy(&err));
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn the_best_checkpoint_survives_cleanup_and_auto_resume_skips_it() {
    let dir = scratch_dir("eval-best");
    let mut cfg = eval_config(&dir);
    cfg.trainer.save_interval = 1;
    cfg.trainer.max_checkpoints = 1;
    let (mut train, evaluator) = eval_loaders(&cfg);
    fit(&cfg, &mut train, Some(evaluator), &mut TuiSink::hidden());

    let prefix = cfg.model.checkpoint_prefix();
    let best = CheckpointLabel::Best;
    for name in [
        best.weights_file_name(&prefix),
        best.optimizer_file_name(&prefix),
        best.meta_file_name(),
    ] {
        assert!(dir.join(&name).exists(), "{name}");
    }
    let best_meta =
        CheckpointMeta::load(&dir.join(best.meta_file_name()).to_string_lossy()).expect("meta");
    assert_eq!(best_meta.evals_since_best, Some(0));
    let records = evals(&dir);
    let best_record = records
        .iter()
        .rev()
        .find(|r| r.since_best == 0)
        .expect("an improving eval");
    assert_eq!(best_meta.step, best_record.step);
    assert_eq!(best_meta.best_eval_loss, Some(best_record.loss));

    remove_old_checkpoints(&dir, 1);
    assert!(dir.join(best.meta_file_name()).exists());
    let (weights, _, step) = find_latest_checkpoint(&cfg.trainer.checkpoint_dir)
        .expect("scan")
        .expect("a step set");
    assert_eq!(step, STEPS);
    assert!(weights.ends_with(&CheckpointLabel::Step(STEPS).weights_file_name(&prefix)));
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn from_yaml_names_the_eval_key_it_refuses() {
    let dir = scratch_dir("eval-validate");
    let mut cfg = tiny_config(&dir, STEPS);
    let mut eval = holdout_eval(HOLDOUT, 1, 2);
    eval.holdout_rows = None;
    cfg.trainer.eval = Some(eval);
    let yaml = dir.join("config.yaml");
    write_config_yaml(&yaml, &cfg);
    let err = LitConfig::from_yaml(&yaml).expect_err("no eval rows");
    assert!(
        err.to_string().contains("trainer.eval.holdout_rows"),
        "{err}"
    );

    cfg.trainer.eval = None;
    write_config_yaml(&yaml, &cfg);
    let text = fs::read_to_string(&yaml).expect("yaml");
    assert!(
        !text.contains("eval:"),
        "an unset eval stays out of the config: {text}"
    );
    let _ = fs::remove_dir_all(dir);
}
