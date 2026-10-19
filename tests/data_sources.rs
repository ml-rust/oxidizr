//! Several token files read as one row stream.
//!
//! - The stream is file 0's whole rows, then file 1's. No row spans two files.
//! - Rank windows over two files concatenate to the global window.
//! - A holdout fraction splits the end of each file. Train and eval share no row.
//! - A resume over two files is bit-identical, and refuses changed sources.
//! - A one-source `trainer.data.sources` trains bit-identical to `-d`.
//! - Sidecars bound the vocabulary and the token count.

mod common;

use boostr::{CpuDevice, CpuRuntime};
use common::{
    assert_same_checkpoints, checkpoint_meta_path, checkpoint_weights, cli_train_args,
    sample_token_meta, scratch_dir, tiny_config, weights_file_bits, write_token_file,
};
use oxidizr::checkpoint_set::CheckpointLabel;
use oxidizr::commands::run_train;
use oxidizr::commands::train_data::{run_loaders, TokenSource};
use oxidizr::config::{CheckpointMeta, LitConfig};
use oxidizr::data::{
    source_records, write_token_meta, Batch, DataConfig, DataSourceSpec, LitDataLoader, RowOrder,
};
use oxidizr::eval::EvalConfig;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

const SEQ_LEN: usize = 3;
const WIDTH: usize = SEQ_LEN + 1;

/// Writes files of `rows[i]` whole rows plus a 2-token tail into `dir`.
///
/// Row tokens are their offset in the concatenation of every file's whole
/// rows, so a row's first token over `WIDTH` names its global row. Tail
/// tokens are `u32::MAX`, so a row that reads a tail or spans two files is
/// caught by [`rows_of`].
fn offset_files(dir: &Path, rows: &[usize]) -> Vec<PathBuf> {
    let mut first = 0u32;
    rows.iter()
        .enumerate()
        .map(|(i, &n)| {
            let mut tokens: Vec<u32> = (first..first + (n * WIDTH) as u32).collect();
            tokens.extend([u32::MAX, u32::MAX]);
            first += (n * WIDTH) as u32;
            let path = dir.join(format!("part{i}.bin"));
            write_token_file(&path, &tokens);
            path
        })
        .collect()
}

/// The global row of each row of `batch`, checking each is `WIDTH` consecutive offsets.
fn rows_of(batch: &Batch<CpuRuntime>) -> Vec<usize> {
    let inputs: Vec<i64> = batch.inputs.contiguous().expect("contiguous").to_vec();
    let targets: Vec<i64> = batch.targets.contiguous().expect("contiguous").to_vec();
    inputs
        .chunks(SEQ_LEN)
        .zip(targets.chunks(SEQ_LEN))
        .map(|(inp, tgt)| {
            let start = inp[0] as usize;
            assert_eq!(start % WIDTH, 0, "row starts at offset {start}");
            let expected: Vec<i64> = (start..start + WIDTH).map(|t| t as i64).collect();
            assert_eq!(
                inp,
                &expected[..SEQ_LEN],
                "a row crossed a file or read a tail"
            );
            assert_eq!(tgt, &expected[1..], "a row crossed a file or read a tail");
            start / WIDTH
        })
        .collect()
}

fn loader(paths: &[PathBuf], batch_size: usize) -> LitDataLoader<CpuRuntime> {
    LitDataLoader::<CpuRuntime>::from_files(paths, batch_size, SEQ_LEN, CpuDevice::new())
        .expect("open files")
}

fn draw(l: &mut LitDataLoader<CpuRuntime>, batches: usize) -> Vec<usize> {
    (0..batches)
        .flat_map(|_| rows_of(&l.next_batch().expect("batch")))
        .collect()
}

#[test]
fn concat_rows_are_each_files_rows_in_order() {
    let dir = scratch_dir("sources-concat");
    let paths = offset_files(&dir, &[5, 3, 4]);
    let mut all = loader(&paths, 1);
    assert_eq!(all.source_rows(), vec![5, 3, 4]);
    assert_eq!(all.total_rows(), 12);
    let expected: Vec<usize> = (0..12).collect();
    assert_eq!(draw(&mut all, 12), expected);

    // Each file alone holds the same rows: its offsets continue the files before it.
    let mut per_file = Vec::new();
    for path in &paths {
        let mut one = loader(std::slice::from_ref(path), 1);
        let n = one.total_rows();
        per_file.extend(draw(&mut one, n));
    }
    assert_eq!(per_file, expected);

    let mut shuffled = loader(&paths, 3).with_row_order(RowOrder::Shuffled { seed: 5 });
    let mut rows = draw(&mut shuffled, 4);
    rows.sort_unstable();
    assert_eq!(rows, expected, "a shuffled epoch reads every row once");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn rank_windows_over_two_files_concatenate_to_the_global_window() {
    let dir = scratch_dir("sources-ranks");
    let paths = offset_files(&dir, &[13, 11]);
    let order = RowOrder::Shuffled { seed: 21 };
    let mut global = loader(&paths, 4).with_row_order(order);
    let mut ranks: Vec<_> = (0..2)
        .map(|rank| {
            loader(&paths, 2)
                .with_row_order(order)
                .with_rank(rank, 2)
                .expect("valid rank")
        })
        .collect();
    let per_epoch = global.row_layout().expect("layout").batches_per_epoch();
    for step in 0..3 * per_epoch + 1 {
        let union: Vec<usize> = ranks
            .iter_mut()
            .flat_map(|l| rows_of(&l.next_batch().expect("rank batch")))
            .collect();
        assert_eq!(union, draw(&mut global, 1), "step {step}");
        for rank in &ranks {
            assert_eq!(rank.position(), global.position(), "step {step}");
        }
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// `tiny_config` with rows of `WIDTH` tokens and an eval of `eval`.
fn holdout_config(dir: &Path, eval: EvalConfig) -> LitConfig {
    let mut cfg = tiny_config(&dir.join("ckpt"), 4);
    cfg.model.max_seq_len = SEQ_LEN;
    cfg.trainer.eval = Some(eval);
    cfg
}

fn fraction_eval(fraction: f64) -> EvalConfig {
    EvalConfig {
        data_file: None,
        holdout_rows: None,
        holdout_fraction: Some(fraction),
        interval: 1,
        batches: 1,
        early_stopping: None,
    }
}

#[test]
fn a_holdout_fraction_splits_the_end_of_each_file_and_train_never_reads_it() {
    let dir = scratch_dir("sources-holdout");
    let paths = offset_files(&dir, &[40, 21]);
    let cfg = holdout_config(&dir, fraction_eval(0.25));
    let files = paths
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    let source = TokenSource::Files(files);
    let loaders = run_loaders::<CpuRuntime>(&cfg, &source, &CpuDevice::new()).expect("loaders");
    let evaluator = loaders.evaluator.expect("an eval");
    assert_eq!(loaders.train.row_ranges(), &[(0, 30), (40, 56)]);
    assert_eq!(evaluator.row_ranges(), &[(30, 40), (56, 61)]);
    assert_eq!(loaders.holdout_rows, 15);
    assert_eq!(evaluator.holdout_rows(), 15);

    let held: HashSet<usize> = evaluator
        .row_ranges()
        .iter()
        .flat_map(|&(s, e)| s..e)
        .collect();
    let mut train = loaders.train.with_row_order(RowOrder::Shuffled { seed: 3 });
    let per_epoch = train.row_layout().expect("layout").batches_per_epoch();
    let seen: HashSet<usize> = draw(&mut train, 2 * per_epoch).into_iter().collect();
    assert!(
        seen.is_disjoint(&held),
        "a training batch read a held-out row"
    );
    assert_eq!(seen.len(), 46, "training reads every training row");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn holdout_rows_over_two_files_is_an_error_naming_holdout_fraction() {
    let dir = scratch_dir("sources-holdout-rows");
    let paths = offset_files(&dir, &[20, 20]);
    let mut eval = fraction_eval(0.1);
    eval.holdout_fraction = None;
    eval.holdout_rows = Some(4);
    let cfg = holdout_config(&dir, eval);
    let files = paths
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    let Err(err) = run_loaders::<CpuRuntime>(&cfg, &TokenSource::Files(files), &CpuDevice::new())
    else {
        panic!("holdout_rows over two files must be refused");
    };
    assert!(
        err.to_string().contains("trainer.eval.holdout_fraction"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// Two token files of `cfg`'s vocabulary with different contents.
fn two_sources(dir: &Path, cfg: &LitConfig) -> (PathBuf, PathBuf) {
    let vocab = cfg.model.vocab_size as u32;
    let a = dir.join("a.bin");
    let b = dir.join("b.bin");
    write_token_file(
        &a,
        &(0..2003u32)
            .map(|i| (i * 7 + i / 5) % vocab)
            .collect::<Vec<_>>(),
    );
    write_token_file(
        &b,
        &(0..1801u32)
            .map(|i| (i * 5 + 3) % vocab)
            .collect::<Vec<_>>(),
    );
    (a, b)
}

/// `tiny_config` of 6 steps saved every 3, its checkpoints in `dir/<ckpt>`.
fn resume_config(dir: &Path, ckpt: &str) -> LitConfig {
    let mut cfg = tiny_config(&dir.join(ckpt), 6);
    cfg.trainer.save_interval = 3;
    cfg
}

#[test]
fn a_two_source_resume_is_bit_identical_and_records_both_sources() {
    let dir = scratch_dir("sources-resume");
    let cfg = resume_config(&dir, "ckpt");
    let (a, b) = two_sources(&dir, &cfg);
    let data = [a.clone(), b.clone()];
    run_train(cli_train_args(&dir, &cfg, &data, "full")).expect("full run");
    let full = weights_file_bits(&cfg, &checkpoint_weights(&cfg, CheckpointLabel::Final));

    let meta_path = checkpoint_meta_path(&cfg, CheckpointLabel::Step(3));
    let meta = CheckpointMeta::load(&meta_path.to_string_lossy()).expect("meta");
    let paths: Vec<String> = meta.data_sources.iter().map(|s| s.path.clone()).collect();
    let expected: Vec<String> = data
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    assert_eq!(paths, expected);
    assert!(meta.data_file_path.is_empty() && meta.data_file_hash.is_empty());
    assert_eq!(meta.data_sources[0].rows, 2003 / 9);
    assert_eq!(meta.data_sources[1].rows, 1801 / 9);

    let mut args = cli_train_args(&dir, &cfg, &data, "resumed");
    args.resume = Some(checkpoint_weights(&cfg, CheckpointLabel::Step(3)));
    run_train(args).expect("resumed run");
    let resumed = weights_file_bits(&cfg, &checkpoint_weights(&cfg, CheckpointLabel::Final));
    assert_eq!(resumed, full, "the two-source resume diverged");

    // The same files in another order are refused.
    let mut args = cli_train_args(&dir, &cfg, &[b, a], "reordered");
    args.resume = Some(checkpoint_weights(&cfg, CheckpointLabel::Step(3)));
    let err = run_train(args).expect_err("reordered sources must be refused");
    let text = format!("{err:#}");
    assert!(text.contains("reordered"), "{text}");
    assert!(text.contains("original order"), "{text}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_one_source_config_trains_bit_identical_to_the_data_flag() {
    let dir = scratch_dir("sources-one");
    let flag_cfg = resume_config(&dir, "ckpt-flag");
    let (a, _) = two_sources(&dir, &flag_cfg);
    run_train(cli_train_args(
        &dir,
        &flag_cfg,
        std::slice::from_ref(&a),
        "flag",
    ))
    .expect("-d run");

    let mut yaml_cfg = resume_config(&dir, "ckpt-yaml");
    yaml_cfg.trainer.data = Some(DataConfig {
        sources: vec![DataSourceSpec {
            path: a.to_string_lossy().into_owned(),
        }],
    });
    run_train(cli_train_args(&dir, &yaml_cfg, &[], "yaml")).expect("yaml run");
    let checked = assert_same_checkpoints(&dir.join("ckpt-flag"), &dir.join("ckpt-yaml"));
    assert!(!checked.is_empty(), "no checkpoint files compared");

    // Both places at once is an error naming both.
    let err = run_train(cli_train_args(&dir, &yaml_cfg, &[a], "both")).expect_err("both set");
    let text = format!("{err:#}");
    assert!(
        text.contains("-d/--data") && text.contains("trainer.data.sources"),
        "{text}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_legacy_one_file_meta_still_resumes_bit_identical() {
    let dir = scratch_dir("sources-legacy");
    let cfg = resume_config(&dir, "ckpt");
    let (a, _) = two_sources(&dir, &cfg);
    let data = [a];
    run_train(cli_train_args(&dir, &cfg, &data, "full")).expect("full run");
    let full = weights_file_bits(&cfg, &checkpoint_weights(&cfg, CheckpointLabel::Final));

    let meta_path = checkpoint_meta_path(&cfg, CheckpointLabel::Step(3));
    let text = std::fs::read_to_string(&meta_path).expect("meta");
    let mut json: serde_json::Value = serde_json::from_str(&text).expect("json");
    let map = json.as_object_mut().expect("meta object");
    assert!(
        map.remove("data_sources").is_some(),
        "the meta lists no sources"
    );
    assert!(!map["data_file_hash"]
        .as_str()
        .unwrap_or_default()
        .is_empty());
    std::fs::write(&meta_path, json.to_string()).expect("write legacy meta");

    let mut args = cli_train_args(&dir, &cfg, &data, "resumed");
    args.resume = Some(checkpoint_weights(&cfg, CheckpointLabel::Step(3)));
    run_train(args).expect("resume of a legacy meta");
    let resumed = weights_file_bits(&cfg, &checkpoint_weights(&cfg, CheckpointLabel::Final));
    assert_eq!(resumed, full, "the legacy meta resumed differently");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_meta_refuses_a_changed_hash_order_or_count() {
    let dir = scratch_dir("sources-meta");
    let cfg = tiny_config(&dir, 2);
    let (a, b) = two_sources(&dir, &cfg);
    let (a, b) = (
        a.to_string_lossy().into_owned(),
        b.to_string_lossy().into_owned(),
    );
    let seq_len = cfg.model.max_seq_len;
    let records = source_records(&[a.clone(), b.clone()], seq_len).expect("records");
    let meta = CheckpointMeta::with_data_hash(2, &cfg.model, &records, "w.safetensors", None);

    let same = meta
        .validate_sources(&cfg.model, &[a.clone(), b.clone()])
        .expect("validate");
    assert!(same.is_valid() && same.data_mismatch.is_none());

    let swapped = meta
        .validate_sources(&cfg.model, &[b.clone(), a.clone()])
        .expect("validate");
    assert!(!swapped.data_matches);
    let text = swapped.data_mismatch.expect("a mismatch");
    assert!(
        text.contains("data source 0") && text.contains("reordered"),
        "{text}"
    );

    let fewer = meta
        .validate_sources(&cfg.model, std::slice::from_ref(&a))
        .expect("validate");
    let text = fewer.data_mismatch.expect("a mismatch");
    assert!(
        text.contains("2 data source(s)") && text.contains("names 1"),
        "{text}"
    );

    let changed = dir.join("b.bin");
    write_token_file(&changed, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let other = meta
        .validate_sources(&cfg.model, &[a.clone(), b.clone()])
        .expect("validate");
    let text = other.data_mismatch.expect("a mismatch");
    assert!(
        text.contains("data source 1") && !text.contains("reordered"),
        "{text}"
    );
    assert!(text.contains("start a fresh run"), "{text}");

    // A meta from before `data_sources` reads as its one data file.
    let mut legacy = CheckpointMeta::new(2, &cfg.model, Some(a.as_str()), "w.safetensors", None)
        .expect("one-file meta");
    legacy.data_sources.clear();
    assert!(legacy
        .validate_sources(&cfg.model, std::slice::from_ref(&a))
        .expect("validate")
        .is_valid());
    let two = legacy
        .validate_sources(&cfg.model, &[a, b])
        .expect("validate");
    assert!(!two.data_matches, "a second source matched a one-file meta");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_sidecar_id_at_or_above_the_vocab_is_an_error() {
    let dir = scratch_dir("sources-vocab");
    let cfg = tiny_config(&dir, 2);
    let vocab = cfg.model.vocab_size as u32;
    let path = dir.join("wide.bin");
    let tokens: Vec<u32> = (0..90).map(|i| i % vocab).collect();
    write_token_file(&path, &tokens);
    write_token_meta(&path, &sample_token_meta(90, Some(vocab))).expect("sidecar");
    let source = TokenSource::Files(vec![path.to_string_lossy().into_owned()]);
    let Err(err) = source.open::<CpuRuntime>(&cfg, &CpuDevice::new()) else {
        panic!("a sidecar id at the vocab size must be refused");
    };
    let text = err.to_string();
    assert!(text.contains("wide.bin"), "{text}");
    assert!(text.contains(&format!("token id {vocab}")), "{text}");
    assert!(
        text.contains(&format!("model.vocab_size is {vocab}")),
        "{text}"
    );
    assert!(text.contains(&format!("at least {}", vocab + 1)), "{text}");

    write_token_meta(&path, &sample_token_meta(90, Some(vocab - 1))).expect("sidecar");
    assert!(source.open::<CpuRuntime>(&cfg, &CpuDevice::new()).is_ok());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_sidecar_token_count_that_disagrees_with_the_file_is_an_error() {
    let dir = scratch_dir("sources-count");
    let path = dir.join("short.bin");
    write_token_file(&path, &[1, 2, 3, 4, 5, 6, 7, 8]);
    write_token_meta(&path, &sample_token_meta(9, Some(8))).expect("sidecar");
    let Err(err) = LitDataLoader::<CpuRuntime>::from_file(&path, 1, SEQ_LEN, CpuDevice::new())
    else {
        panic!("a sidecar of 9 tokens over a file of 8 must be refused");
    };
    let text = err.to_string();
    assert!(
        text.contains("num_tokens 9") && text.contains("holds 8"),
        "{text}"
    );
    assert!(text.contains("short.bin.tokens.json"), "{text}");
    let _ = std::fs::remove_dir_all(dir);
}
