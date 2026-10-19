//! A weighted mixture of token files, read as a virtual row order.
//!
//! - Each epoch reads each source's largest-remainder quota exactly.
//! - A seek mid-epoch continues the uninterrupted rows, across pass and epoch boundaries.
//! - Rank windows concatenate to the global window.
//! - Stream order is fixed. A seed fixes the shuffle, and each epoch draws another.
//! - The holdout comes off each source before the mixture. The eval stays unmixed.
//! - A resume under a mixture is bit-identical. A resume with another mixture is an error.
//! - Without weights a run trains as before. Bad weights are errors naming the key.

mod common;

use boostr::{CpuDevice, CpuRuntime};
use common::{
    assert_same_checkpoints, checkpoint_meta_path, checkpoint_weights, cli_train_args, offset_rows,
    scratch_dir, tiny_config, weights_file_bits, write_offset_files, write_token_file,
};
use oxidizr::checkpoint_set::CheckpointLabel;
use oxidizr::commands::run_train;
use oxidizr::commands::train_data::{run_loaders, TokenSource};
use oxidizr::config::{CheckpointMeta, LitConfig};
use oxidizr::data::{DataConfig, DataPosition, DataSourceSpec, LitDataLoader, RowOrder};
use oxidizr::eval::EvalConfig;
use std::path::{Path, PathBuf};

const SEQ_LEN: usize = 3;

/// A loader over `paths` at `batch_size` rows, mixed under `weights`.
fn mixed(
    paths: &[PathBuf],
    batch_size: usize,
    weights: &[f64],
    rows_per_epoch: Option<usize>,
) -> LitDataLoader<CpuRuntime> {
    LitDataLoader::<CpuRuntime>::from_files(paths, batch_size, SEQ_LEN, CpuDevice::new())
        .expect("open files")
        .with_mixture(weights, rows_per_epoch)
        .expect("mixture")
}

fn draw(l: &mut LitDataLoader<CpuRuntime>, batches: usize) -> Vec<usize> {
    (0..batches)
        .flat_map(|_| offset_rows(&l.next_batch().expect("batch"), SEQ_LEN))
        .collect()
}

/// The source of global row `row` over files of `rows[i]` rows.
fn source_of(rows: &[usize], row: usize) -> usize {
    let mut end = 0;
    rows.iter()
        .position(|&n| {
            end += n;
            row < end
        })
        .expect("a row inside the files")
}

/// Rows each source supplies to `rows`.
fn per_source(sizes: &[usize], rows: &[usize]) -> Vec<usize> {
    let mut counts = vec![0; sizes.len()];
    for &row in rows {
        counts[source_of(sizes, row)] += 1;
    }
    counts
}

fn paths_of(paths: &[PathBuf]) -> Vec<String> {
    paths
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect()
}

#[test]
fn each_epoch_reads_each_quota_exactly() {
    let dir = scratch_dir("mixture-quota");
    let sizes = [10, 3, 7];
    let paths = write_offset_files(&dir, &sizes, SEQ_LEN);
    // Shares 10/3 each: one extra row, to source 0 on the tie.
    let cases = [
        ([1.0, 1.0, 1.0], 10, [4, 3, 3]),
        ([1.0, 2.0, 1.0], 8, [2, 4, 2]),
    ];
    for (weights, rows, want) in cases {
        for order in [RowOrder::Stream, RowOrder::Shuffled { seed: 4 }] {
            let mut l = mixed(&paths, 1, &weights, Some(rows)).with_row_order(order);
            let quotas: Vec<usize> = l
                .mixture_sources()
                .expect("sources")
                .iter()
                .map(|s| s.rows_per_epoch)
                .collect();
            assert_eq!(quotas, want);
            assert_eq!(quotas.iter().sum::<usize>(), rows);
            for epoch in 0..3 {
                let got = per_source(&sizes, &draw(&mut l, rows));
                assert_eq!(got, want.to_vec(), "{order:?} epoch {epoch}");
            }
        }
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_seek_continues_the_uninterrupted_rows_across_passes_and_epochs() {
    let dir = scratch_dir("mixture-seek");
    // Source 1 holds 3 rows and supplies 8 of each 16-row epoch: almost 3 passes per epoch.
    let paths = write_offset_files(&dir, &[20, 3], SEQ_LEN);
    let make =
        || mixed(&paths, 3, &[1.0, 1.0], Some(16)).with_row_order(RowOrder::Shuffled { seed: 8 });
    let mut full = make();
    let layout = full.row_layout().expect("layout");
    assert_eq!(layout.batches_per_epoch(), 5);
    let reference = draw(&mut full, 18);
    for stop in 0..=12 {
        let mut resumed = make();
        let position = layout.advance(DataPosition::default(), stop);
        resumed.seek(position).expect("seek");
        assert_eq!(
            draw(&mut resumed, 6),
            reference[stop * 3..(stop + 6) * 3],
            "stopped after {stop} batches at {position:?}"
        );
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn rank_windows_concatenate_to_the_global_window() {
    let dir = scratch_dir("mixture-ranks");
    let paths = write_offset_files(&dir, &[13, 4, 9], SEQ_LEN);
    let (weights, rows) = ([2.0, 1.0, 1.0], Some(22));
    let order = RowOrder::Shuffled { seed: 21 };
    let mut global = mixed(&paths, 4, &weights, rows).with_row_order(order);
    let mut ranks: Vec<_> = (0..2)
        .map(|rank| {
            mixed(&paths, 2, &weights, rows)
                .with_row_order(order)
                .with_rank(rank, 2)
                .expect("valid rank")
        })
        .collect();
    let per_epoch = global.row_layout().expect("layout").batches_per_epoch();
    for step in 0..3 * per_epoch + 1 {
        let union: Vec<usize> = ranks
            .iter_mut()
            .flat_map(|l| offset_rows(&l.next_batch().expect("rank batch"), SEQ_LEN))
            .collect();
        assert_eq!(union, draw(&mut global, 1), "step {step}");
        for rank in &ranks {
            assert_eq!(rank.position(), global.position(), "step {step}");
        }
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn stream_order_interleaves_the_same_way_every_epoch() {
    let dir = scratch_dir("mixture-stream");
    let sizes = [6, 4];
    let paths = write_offset_files(&dir, &sizes, SEQ_LEN);
    let make = || mixed(&paths, 1, &[3.0, 1.0], Some(8)).with_row_order(RowOrder::Stream);
    let (mut a, mut b) = (make(), make());
    let rows = draw(&mut a, 24);
    assert_eq!(draw(&mut b, 24), rows, "stream order is deterministic");
    let sources: Vec<usize> = rows.iter().map(|&r| source_of(&sizes, r)).collect();
    assert_eq!(
        sources[..8],
        sources[8..16],
        "every epoch interleaves the same way"
    );
    assert_eq!(sources[..8], [0, 0, 1, 0, 0, 0, 1, 0]);
    // Each source reads its rows in order, and wraps at its end.
    let first: Vec<usize> = rows.iter().copied().filter(|&r| r < 6).collect();
    let want: Vec<usize> = (0..6).cycle().take(first.len()).collect();
    assert_eq!(first, want);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_seed_fixes_the_shuffle_and_each_epoch_draws_another() {
    let dir = scratch_dir("mixture-seed");
    let sizes = [30, 30];
    let paths = write_offset_files(&dir, &sizes, SEQ_LEN);
    let make =
        |seed| mixed(&paths, 4, &[1.0, 1.0], None).with_row_order(RowOrder::Shuffled { seed });
    let (mut a, mut b) = (make(5), make(5));
    let (a0, a1) = (draw(&mut a, 15), draw(&mut a, 15));
    assert_eq!(draw(&mut b, 15), a0, "same seed, same epoch 0");
    assert_eq!(draw(&mut b, 15), a1, "same seed, same epoch 1");
    assert_ne!(a0, a1, "each epoch draws another order");
    let schedule =
        |rows: &[usize]| -> Vec<usize> { rows.iter().map(|&r| source_of(&sizes, r)).collect() };
    assert_ne!(
        schedule(&a0),
        schedule(&a1),
        "each epoch draws another schedule"
    );
    assert_ne!(draw(&mut make(6), 15), a0, "another seed, another order");
    let _ = std::fs::remove_dir_all(dir);
}

/// `trainer.data` over `paths` with `weights`, one per source, or none.
fn data_config(paths: &[PathBuf], weights: Option<&[f64]>, rows: Option<usize>) -> DataConfig {
    DataConfig {
        sources: paths
            .iter()
            .enumerate()
            .map(|(i, p)| DataSourceSpec {
                path: p.to_string_lossy().into_owned(),
                weight: weights.map(|w| w[i]),
            })
            .collect(),
        rows_per_epoch: rows,
    }
}

#[test]
fn the_holdout_comes_off_each_source_before_the_mixture() {
    let dir = scratch_dir("mixture-holdout");
    let sizes = [40, 20];
    let paths = write_offset_files(&dir, &sizes, SEQ_LEN);
    let mut cfg = tiny_config(&dir.join("ckpt"), 4);
    cfg.model.max_seq_len = SEQ_LEN;
    cfg.trainer.data = Some(data_config(&paths, Some(&[1.0, 3.0][..]), None));
    cfg.trainer.eval = Some(EvalConfig {
        data_file: None,
        holdout_rows: None,
        holdout_fraction: Some(0.25),
        interval: 1,
        batches: 1,
        early_stopping: None,
    });
    let source = TokenSource::Files(paths_of(&paths));
    let loaders = run_loaders::<CpuRuntime>(&cfg, &source, &CpuDevice::new()).expect("loaders");
    let evaluator = loaders.evaluator.expect("an eval");
    assert_eq!(
        evaluator.row_ranges(),
        &[(30, 40), (55, 60)],
        "the eval is unmixed"
    );
    let sources = loaders.train.mixture_sources().expect("a mixture");
    let train_rows: Vec<usize> = sources.iter().map(|s| s.train_rows).collect();
    assert_eq!(train_rows, vec![30, 15]);
    // 45 rows per epoch: shares 11.25 and 33.75.
    let quotas: Vec<usize> = sources.iter().map(|s| s.rows_per_epoch).collect();
    assert_eq!(quotas, vec![11, 34]);
    let mut train = loaders.train.with_row_order(RowOrder::Shuffled { seed: 3 });
    let rows = draw(&mut train, 3 * 22);
    let held = |r: &usize| (30..40).contains(r) || (55..60).contains(r);
    assert!(
        !rows.iter().any(held),
        "a training batch read a held-out row"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// Token files of `cfg`'s vocabulary: two large sources and one of 3 rows.
fn three_sources(dir: &Path, cfg: &LitConfig) -> Vec<PathBuf> {
    let vocab = cfg.model.vocab_size as u32;
    let row = cfg.model.max_seq_len as u32 + 1;
    let specs = [(2003, 7, 1), (1801, 5, 3), (3 * row + 2, 3, 2)];
    specs
        .iter()
        .enumerate()
        .map(|(i, &(len, mul, add))| {
            let path = dir.join(format!("src{i}.bin"));
            let tokens: Vec<u32> = (0..len).map(|t| (t * mul + add + t / 5) % vocab).collect();
            write_token_file(&path, &tokens);
            path
        })
        .collect()
}

/// `tiny_config` of 6 steps saved every 3, its checkpoints in `dir/<ckpt>`.
///
/// Epochs of 16 rows are 4 steps, so a resume at step 3 crosses an epoch.
fn mixture_config(dir: &Path, ckpt: &str, data: DataConfig) -> LitConfig {
    let mut cfg = tiny_config(&dir.join(ckpt), 6);
    cfg.trainer.save_interval = 3;
    cfg.trainer.num_epochs = 4;
    cfg.trainer.data = Some(data);
    cfg
}

/// Resumes `cfg` from its step-3 checkpoint as run `run`.
fn resume_from_step_3(dir: &Path, cfg: &LitConfig, run: &str) -> oxidizr::prelude::Result<()> {
    let mut args = cli_train_args(dir, cfg, &[], run);
    args.resume = Some(checkpoint_weights(cfg, CheckpointLabel::Step(3)));
    run_train(args).map_err(|e| oxidizr::prelude::Error::TrainingError {
        reason: format!("{e:#}"),
    })
}

#[test]
fn a_mixture_resume_is_bit_identical_and_refuses_another_mixture() {
    let dir = scratch_dir("mixture-resume");
    let probe = tiny_config(&dir, 1);
    let paths = three_sources(&dir, &probe);
    let weights = [1.0, 1.0, 2.0];
    let cfg = mixture_config(
        &dir,
        "ckpt",
        data_config(&paths, Some(&weights[..]), Some(16)),
    );
    run_train(cli_train_args(&dir, &cfg, &[], "full")).expect("full run");
    let full = weights_file_bits(&cfg, &checkpoint_weights(&cfg, CheckpointLabel::Final));

    let meta_path = checkpoint_meta_path(&cfg, CheckpointLabel::Step(3));
    let meta = CheckpointMeta::load(&meta_path.to_string_lossy()).expect("meta");
    let mixture = meta.mixture.expect("the meta records the mixture");
    assert_eq!(mixture.weights, weights.to_vec());
    assert_eq!(mixture.rows_per_epoch, 16);

    resume_from_step_3(&dir, &cfg, "resumed").expect("resumed run");
    let resumed = weights_file_bits(&cfg, &checkpoint_weights(&cfg, CheckpointLabel::Final));
    assert_eq!(resumed, full, "the mixture resume diverged");

    let changed = [
        (
            Some(&[1.0, 1.0, 3.0][..]),
            Some(16),
            "trainer.data.sources[2].weight was 2",
        ),
        (
            Some(&weights[..]),
            Some(20),
            "trainer.data.rows_per_epoch was 16",
        ),
        (None, None, "this run sets no weights"),
    ];
    for (i, (w, rows, want)) in changed.into_iter().enumerate() {
        let other = mixture_config(&dir, "ckpt", data_config(&paths, w, rows));
        let err = resume_from_step_3(&dir, &other, &format!("changed{i}")).expect_err(want);
        let text = err.to_string();
        assert!(text.contains(want), "{text}");
        assert!(text.contains("start a fresh run"), "{text}");
    }

    // A meta without a mixture resumes only a run without weights.
    let text = std::fs::read_to_string(&meta_path).expect("meta");
    let mut json: serde_json::Value = serde_json::from_str(&text).expect("json");
    let map = json.as_object_mut().expect("meta object");
    assert!(map.remove("mixture").is_some());
    std::fs::write(&meta_path, json.to_string()).expect("write meta");
    let err = resume_from_step_3(&dir, &cfg, "legacy").expect_err("a meta without a mixture");
    assert!(err.to_string().contains("trained without"), "{err}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_config_without_weights_trains_as_the_data_flags_do() {
    let dir = scratch_dir("mixture-none");
    let probe = tiny_config(&dir, 1);
    let paths = three_sources(&dir, &probe)[..2].to_vec();
    let mut flag_cfg = tiny_config(&dir.join("ckpt-flag"), 6);
    flag_cfg.trainer.save_interval = 3;
    run_train(cli_train_args(&dir, &flag_cfg, &paths, "flag")).expect("-d run");
    let mut plain = tiny_config(&dir.join("ckpt-yaml"), 6);
    plain.trainer.save_interval = 3;
    plain.trainer.data = Some(data_config(&paths, None, None));
    run_train(cli_train_args(&dir, &plain, &[], "yaml")).expect("yaml run");
    let checked = assert_same_checkpoints(&dir.join("ckpt-flag"), &dir.join("ckpt-yaml"));
    assert!(!checked.is_empty(), "no checkpoint files compared");
    let meta = std::fs::read_to_string(checkpoint_meta_path(&plain, CheckpointLabel::Step(3)))
        .expect("meta");
    assert!(
        !meta.contains("\"mixture\""),
        "a plain run recorded a mixture"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn one_weighted_source_trains_the_weights_of_the_data_flag() {
    let dir = scratch_dir("mixture-one");
    let probe = tiny_config(&dir, 1);
    let path = three_sources(&dir, &probe).swap_remove(0);
    let flag_cfg = tiny_config(&dir.join("ckpt-flag"), 6);
    run_train(cli_train_args(
        &dir,
        &flag_cfg,
        std::slice::from_ref(&path),
        "flag",
    ))
    .expect("-d run");
    let flag = weights_file_bits(
        &flag_cfg,
        &checkpoint_weights(&flag_cfg, CheckpointLabel::Final),
    );
    let mut one = tiny_config(&dir.join("ckpt-one"), 6);
    one.trainer.data = Some(data_config(
        std::slice::from_ref(&path),
        Some(&[0.4][..]),
        None,
    ));
    run_train(cli_train_args(&dir, &one, &[], "one")).expect("weighted run");
    let weighted = weights_file_bits(&one, &checkpoint_weights(&one, CheckpointLabel::Final));
    assert_eq!(weighted, flag, "one weighted source trained differently");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn bad_weights_are_errors_naming_the_key() {
    let dir = scratch_dir("mixture-bad");
    let probe = tiny_config(&dir, 1);
    let paths = three_sources(&dir, &probe)[..2].to_vec();
    let cases = [
        (
            Some(&[1.0, 0.0][..]),
            None,
            "trainer.data.sources[1].weight is 0",
        ),
        (
            Some(&[-2.0, 1.0][..]),
            None,
            "trainer.data.sources[0].weight is -2",
        ),
        (None, Some(10), "no source sets a weight"),
        (
            Some(&[1.0, 1.0][..]),
            Some(0),
            "trainer.data.rows_per_epoch is 0",
        ),
    ];
    for (i, (weights, rows, want)) in cases.into_iter().enumerate() {
        let cfg = mixture_config(&dir, "ckpt", data_config(&paths, weights, rows));
        let err = run_train(cli_train_args(&dir, &cfg, &[], &format!("bad{i}"))).expect_err(want);
        let text = format!("{err:#}");
        assert!(text.contains(want), "{text}");
    }
    let mut partial = data_config(&paths, Some(&[1.0, 1.0][..]), None);
    partial.sources[0].weight = None;
    let text = partial.validate().expect_err("partial weights").to_string();
    assert!(
        text.contains("trainer.data.sources[1].weight is set")
            && text.contains("trainer.data.sources[0].weight is not"),
        "{text}"
    );
    let _ = std::fs::remove_dir_all(dir);
}
