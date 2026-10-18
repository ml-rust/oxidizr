//! Data-parallel runs through the whole `train` command path, on in-process CPU ranks.
//!
//! `run_train_with_group` runs every rank of a `RankGroup::local` group on its
//! own thread: its own CPU client, model, loader and trainer. Rank 0 writes the
//! run directory and the checkpoints. These tests cover what one process with
//! two GPUs runs, except NCCL and the CUDA devices.

mod common;

use boostr::runtime::{Communicator, LocalCommunicator, ReduceOp};
use boostr::{CpuRuntime, DType, NumrError, NumrResult};
use common::{
    bits_by_name, build_model, event_files, holdout_eval, scratch_dir, tiny_config, tokens,
    train_parts, write_config_yaml, write_token_file,
};
use oxidizr::checkpoint_set::CheckpointLabel;
use oxidizr::commands::{run_train, run_train_with_group, TrainArgs};
use oxidizr::config::{CheckpointMeta, LitConfig, TargetMaskSpec};
use oxidizr::dp::RankGroup;
use oxidizr::eval::EarlyStoppingConfig;
use oxidizr::model::load_checkpoint_into_model;
use oxidizr::run_record::{RunRecord, RunStatus, StepRecord, EVALS_FILE, STEPS_FILE};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const WORLD: usize = 2;
/// Group timeout of the runs that must finish: long enough never to fire.
const TIMEOUT: Duration = Duration::from_secs(120);

type Bits = BTreeMap<String, Vec<u32>>;

/// Headless CPU args that run `cfg` on `data` and write the run directory `dir/<run>`.
fn args_for(dir: &Path, data: &Path, cfg: &LitConfig, run: &str) -> TrainArgs {
    let cfg_path = dir.join(format!("{run}.yaml"));
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
        dtype: None,
        checkpoint_dir: None,
        run_dir: Some(dir.join(run).to_string_lossy().into_owned()),
    }
}

/// A scratch dir, its token file, and `tiny_config` of `steps` steps logged every step.
///
/// The checkpoints go to `dir/ckpt`, every `save` steps.
fn setup(tag: &str, steps: usize, save: usize) -> (PathBuf, PathBuf, LitConfig) {
    let dir = scratch_dir(tag);
    let mut cfg = tiny_config(&dir.join("ckpt"), steps);
    cfg.trainer.log_interval = 1;
    cfg.trainer.save_interval = save;
    let data = dir.join("tokens.bin");
    write_token_file(&data, &tokens(&cfg, 4000));
    (dir, data, cfg)
}

/// `args` on `WORLD` in-process CPU ranks.
fn run_world(args: TrainArgs) -> anyhow::Result<()> {
    let group = RankGroup::local(WORLD, TIMEOUT).expect("local group");
    run_train_with_group::<CpuRuntime>(args, group)
}

/// Every `steps.json` line of the run directory `run_dir`.
fn step_records(run_dir: &Path) -> Vec<StepRecord> {
    std::fs::read_to_string(run_dir.join(STEPS_FILE))
        .expect("steps.json")
        .lines()
        .map(|line| serde_json::from_str(line).expect("parse step line"))
        .collect()
}

/// The checkpoint file of `label` in `cfg`'s checkpoint directory.
fn weights(cfg: &LitConfig, label: CheckpointLabel) -> String {
    let name = label.weights_file_name(&cfg.model.checkpoint_prefix());
    Path::new(&cfg.trainer.checkpoint_dir)
        .join(name)
        .to_string_lossy()
        .into_owned()
}

/// The meta of `label` in `cfg`'s checkpoint directory.
fn meta(cfg: &LitConfig, label: CheckpointLabel) -> CheckpointMeta {
    let path = Path::new(&cfg.trainer.checkpoint_dir).join(label.meta_file_name());
    CheckpointMeta::load(&path.to_string_lossy()).expect("checkpoint meta")
}

/// Bit patterns of every trainable tensor in the weights file at `path`.
fn file_bits(cfg: &LitConfig, path: &str) -> Bits {
    let mut model = build_model(cfg);
    load_checkpoint_into_model(&mut model, path).expect("load weights");
    bits_by_name(&train_parts(&model))
}

/// Sorted entry names of `dir`.
///
/// TensorBoard event files are left out: only a `tensorboard` build writes them.
fn entries(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("read dir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| !name.starts_with("events.out.tfevents."))
        .collect();
    names.sort();
    names
}

/// Every element of `got` is within 1e-6 absolute plus 1e-5 relative of `want`.
fn assert_close(got: &Bits, want: &Bits) {
    assert_eq!(got.len(), want.len());
    for (name, want) in want {
        let got = &got[name];
        assert_eq!(got.len(), want.len(), "{name} length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let (g, w) = (f32::from_bits(*g), f32::from_bits(*w));
            assert!(
                (g - w).abs() <= 1e-6 + 1e-5 * w.abs(),
                "{name}[{i}]: data-parallel {g} != single-rank {w}"
            );
        }
    }
}

#[test]
fn a_world_two_run_writes_rank_zero_outputs_and_matches_one_rank_on_the_merged_batch() {
    let (dir, data, cfg) = setup("dp-run-outputs", 4, 2);
    run_world(args_for(&dir, &data, &cfg, "dp")).expect("world-2 run");

    let run_dir = dir.join("dp");
    assert_eq!(entries(&run_dir), vec!["run.json", "steps.json"]);
    let events = usize::from(cfg!(feature = "tensorboard"));
    assert_eq!(
        event_files(&run_dir).len(),
        events,
        "rank 0 alone writes events"
    );
    let record = RunRecord::load(&run_dir).expect("run.json");
    assert_eq!(record.status, RunStatus::Completed, "{record:?}");
    assert_eq!(record.final_step, 4);
    assert!(record.last_loss.is_some_and(f64::is_finite), "{record:?}");
    let trainer = &record.config.as_ref().expect("config")["trainer"];
    assert_eq!(trainer["world_size"], 2);
    assert_eq!(trainer["batch_size"], 2);
    // 2 rows per rank x 2 ranks x 2 micro-batches.
    assert_eq!(trainer["effective_batch_size"], 8);
    let steps: Vec<usize> = step_records(&run_dir).iter().map(|r| r.step).collect();
    assert_eq!(steps, vec![1, 2, 3, 4]);
    for label in [
        CheckpointLabel::Step(2),
        CheckpointLabel::Step(4),
        CheckpointLabel::Final,
    ] {
        let m = meta(&cfg, label);
        assert_eq!(m.world_size, Some(2), "{label:?}");
        assert_eq!(m.global_batch_rows, Some(4), "{label:?}");
    }

    // One rank whose batch holds both ranks' rows: the same rows per step.
    let mut single = cfg.clone();
    single.trainer.batch_size *= WORLD;
    single.trainer.checkpoint_dir = dir.join("ckpt-single").to_string_lossy().into_owned();
    run_train(args_for(&dir, &data, &single, "single")).expect("world-1 run");
    assert_eq!(meta(&single, CheckpointLabel::Final).world_size, Some(1));
    let dp = file_bits(&cfg, &weights(&cfg, CheckpointLabel::Final));
    let one = file_bits(&single, &weights(&single, CheckpointLabel::Final));
    assert_close(&dp, &one);
    assert_ne!(
        dp,
        bits_by_name(&train_parts(&build_model(&cfg))),
        "nothing trained"
    );
    let dp_steps = step_records(&run_dir);
    for (got, want) in dp_steps.iter().zip(step_records(&dir.join("single"))) {
        let tolerance = 1e-6 * want.loss.abs();
        assert!(
            (got.loss - want.loss).abs() <= tolerance,
            "step {}: loss {} != single-rank {}",
            got.step,
            got.loss,
            want.loss
        );
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_world_two_resume_is_bit_identical_to_an_uninterrupted_run() {
    let (dir, data, cfg) = setup("dp-run-resume", 6, 3);
    run_world(args_for(&dir, &data, &cfg, "full")).expect("full run");
    let full_bits = file_bits(&cfg, &weights(&cfg, CheckpointLabel::Final));
    let full_steps = step_records(&dir.join("full"));

    let mut args = args_for(&dir, &data, &cfg, "resumed");
    args.resume = Some(weights(&cfg, CheckpointLabel::Step(3)));
    run_world(args).expect("resumed run");

    let resumed_bits = file_bits(&cfg, &weights(&cfg, CheckpointLabel::Final));
    assert_eq!(resumed_bits, full_bits, "the resume diverged");
    let resumed_steps = step_records(&dir.join("resumed"));
    let steps: Vec<usize> = resumed_steps.iter().map(|r| r.step).collect();
    assert_eq!(steps, vec![4, 5, 6]);
    for record in &resumed_steps {
        let reference = &full_steps[record.step - 1];
        let bits = |r: &StepRecord| {
            (
                r.loss.to_bits(),
                r.grad_norm.to_bits(),
                r.learning_rate.to_bits(),
            )
        };
        assert_eq!(bits(record), bits(reference), "step {}", record.step);
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_world_two_checkpoint_does_not_resume_at_world_one() {
    let (dir, data, cfg) = setup("dp-run-world-mismatch", 2, 1);
    run_world(args_for(&dir, &data, &cfg, "dp")).expect("world-2 run");

    let mut args = args_for(&dir, &data, &cfg, "one");
    args.resume = Some("auto".to_string());
    let err = format!(
        "{:#}",
        run_train(args).expect_err("world 2 resumed at world 1")
    );
    assert!(err.contains("world size 2"), "{err}");
    assert!(err.contains("world size 1"), "{err}");
    assert!(err.contains("--gpus"), "{err}");
    let record = RunRecord::load(&dir.join("one")).expect("run.json");
    assert_eq!(record.status, RunStatus::Failed);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn an_old_meta_without_world_keys_resumes_at_world_one_unchanged() {
    let (dir, data, cfg) = setup("dp-run-old-meta", 6, 3);
    run_train(args_for(&dir, &data, &cfg, "full")).expect("full run");
    let full_bits = file_bits(&cfg, &weights(&cfg, CheckpointLabel::Final));

    let meta_path = Path::new(&cfg.trainer.checkpoint_dir).join("checkpoint_step_3.meta.json");
    let mut json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&meta_path).expect("meta")).expect("json");
    let map = json.as_object_mut().expect("meta object");
    assert!(
        map.remove("world_size").is_some(),
        "the meta has no world_size"
    );
    assert!(map.remove("global_batch_rows").is_some());
    std::fs::write(&meta_path, json.to_string()).expect("write old meta");

    let mut args = args_for(&dir, &data, &cfg, "resumed");
    args.resume = Some(weights(&cfg, CheckpointLabel::Step(3)));
    run_train(args).expect("resume of an old meta");
    let resumed = file_bits(&cfg, &weights(&cfg, CheckpointLabel::Final));
    assert_eq!(resumed, full_bits, "the old meta resumed differently");
    let _ = std::fs::remove_dir_all(dir);
}

/// What a faulty rank does at its fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fault {
    Error,
    Panic,
}

/// A rank that fails at the step-count sum of step `at`.
///
/// The step weighting is the only `I64` sum of 2 elements, one per step.
struct FaultyRank {
    inner: Arc<LocalCommunicator>,
    fault: Fault,
    at: usize,
    seen: AtomicUsize,
}

impl Communicator for FaultyRank {
    fn world_size(&self) -> usize {
        self.inner.world_size()
    }

    fn rank(&self) -> usize {
        self.inner.rank()
    }

    unsafe fn all_reduce(
        &self,
        ptr: u64,
        count: usize,
        dtype: DType,
        op: ReduceOp,
    ) -> NumrResult<()> {
        if dtype == DType::I64 && count == 2 {
            let step = self.seen.fetch_add(1, Ordering::SeqCst) + 1;
            if step == self.at {
                match self.fault {
                    Fault::Error => {
                        return Err(NumrError::Backend(format!(
                            "injected failure at step {step}"
                        )))
                    }
                    Fault::Panic => panic!("injected panic at step {step}"),
                }
            }
        }
        // SAFETY: forwarded from the caller's contract.
        unsafe { self.inner.all_reduce(ptr, count, dtype, op) }
    }

    unsafe fn broadcast(
        &self,
        ptr: u64,
        count: usize,
        dtype: DType,
        root: usize,
    ) -> NumrResult<()> {
        // SAFETY: forwarded from the caller's contract.
        unsafe { self.inner.broadcast(ptr, count, dtype, root) }
    }

    unsafe fn all_gather(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DType,
    ) -> NumrResult<()> {
        // SAFETY: forwarded from the caller's contract.
        unsafe { self.inner.all_gather(send_ptr, recv_ptr, count, dtype) }
    }

    unsafe fn reduce_scatter(
        &self,
        send_ptr: u64,
        recv_ptr: u64,
        count: usize,
        dtype: DType,
        op: ReduceOp,
    ) -> NumrResult<()> {
        // SAFETY: forwarded from the caller's contract.
        unsafe {
            self.inner
                .reduce_scatter(send_ptr, recv_ptr, count, dtype, op)
        }
    }

    unsafe fn send(
        &self,
        ptr: u64,
        count: usize,
        dtype: DType,
        dest: usize,
        tag: u32,
    ) -> NumrResult<()> {
        // SAFETY: forwarded from the caller's contract.
        unsafe { self.inner.send(ptr, count, dtype, dest, tag) }
    }

    unsafe fn recv(
        &self,
        ptr: u64,
        count: usize,
        dtype: DType,
        src: usize,
        tag: u32,
    ) -> NumrResult<()> {
        // SAFETY: forwarded from the caller's contract.
        unsafe { self.inner.recv(ptr, count, dtype, src, tag) }
    }

    fn sync(&self) -> NumrResult<()> {
        self.inner.sync()
    }

    fn barrier(&self) -> NumrResult<()> {
        self.inner.barrier()
    }
}

/// A world-2 run whose rank 1 hits `fault` at step 2, with its error and duration.
fn faulty_run(tag: &str, fault: Fault, timeout: Duration) -> (String, RunRecord, Duration) {
    let (dir, data, cfg) = setup(tag, 4, 0);
    let locals: Vec<Arc<LocalCommunicator>> = LocalCommunicator::with_timeout(WORLD, timeout)
        .into_iter()
        .map(Arc::new)
        .collect();
    let first = Arc::clone(&locals[0]);
    let comms: Vec<Arc<dyn Communicator>> = vec![
        Arc::clone(&locals[0]) as Arc<dyn Communicator>,
        Arc::new(FaultyRank {
            inner: Arc::clone(&locals[1]),
            fault,
            at: 2,
            seen: AtomicUsize::new(0),
        }),
    ];
    let group = RankGroup::new(comms, Arc::new(move || first.abort())).expect("group");
    let start = Instant::now();
    let result = run_train_with_group::<CpuRuntime>(args_for(&dir, &data, &cfg, "run"), group);
    let elapsed = start.elapsed();
    let err = format!("{:#}", result.expect_err("rank 1 fails"));
    let record = RunRecord::load(&dir.join("run")).expect("run.json after a failed run");
    let _ = std::fs::remove_dir_all(dir);
    (err, record, elapsed)
}

/// Rank 1 fails at step 2. The abort wakes rank 0, which waits in the step-count
/// sum, long before the group timeout. Rank 1's error is the run's error.
#[test]
fn a_rank_failing_at_step_two_fails_the_run_without_a_hang() {
    let timeout = Duration::from_secs(60);
    let (err, record, elapsed) = faulty_run("dp-run-rank-error", Fault::Error, timeout);
    assert!(
        elapsed < timeout,
        "rank 0 waited for the timeout: {elapsed:?}"
    );
    assert!(err.contains("rank 1 of 2 failed first"), "{err}");
    assert!(err.contains("injected failure at step 2"), "{err}");
    assert_eq!(record.status, RunStatus::Failed);
    assert!(
        record
            .error
            .as_deref()
            .is_some_and(|e| e.contains("injected failure")),
        "{record:?}"
    );
    // Rank 0 completed step 1 and failed in step 2.
    assert_eq!(record.final_step, 1);
}

#[test]
fn a_rank_panic_is_the_run_error() {
    let timeout = Duration::from_secs(60);
    let (err, record, elapsed) = faulty_run("dp-run-rank-panic", Fault::Panic, timeout);
    assert!(
        elapsed < timeout,
        "rank 0 waited for the timeout: {elapsed:?}"
    );
    assert!(err.contains("rank 1 of 2 failed first"), "{err}");
    assert!(err.contains("injected panic at step 2"), "{err}");
    assert_eq!(record.status, RunStatus::Failed);
}

/// Rank 0 evaluates. No eval improves after the first, so a patience of 2
/// stops the run after step 3. Rank 1 learns the stop by broadcast. A rank
/// that ran on would mismatch the final barrier and fail the run.
#[test]
fn evals_at_world_two_stop_every_rank_at_the_same_step() {
    let (dir, data, mut cfg) = setup("dp-run-evals", 6, 0);
    let mut eval = holdout_eval(40, 1, 2);
    eval.early_stopping = Some(EarlyStoppingConfig {
        patience: 2,
        min_delta: 1e9,
    });
    cfg.trainer.eval = Some(eval);
    run_world(args_for(&dir, &data, &cfg, "dp")).expect("world-2 run with evals");

    let run_dir = dir.join("dp");
    let record = RunRecord::load(&run_dir).expect("run.json");
    assert_eq!(record.status, RunStatus::Completed, "{record:?}");
    assert!(record.stopped_early, "{record:?}");
    assert_eq!(record.final_step, 3);
    assert_eq!(record.evals, Some(3));
    assert!(record.best_eval_loss.is_some_and(f64::is_finite));
    let evals = std::fs::read_to_string(run_dir.join(EVALS_FILE)).expect("evals.json");
    assert_eq!(evals.lines().count(), 3);
    let final_meta = meta(&cfg, CheckpointLabel::Final);
    assert_eq!((final_meta.step, final_meta.world_size), (3, Some(2)));
    let _ = std::fs::remove_dir_all(dir);
}

/// Every rank masks the same way, so the step weighting sums the scored
/// counts over both ranks, and the run completes.
#[test]
fn a_masked_world_two_run_completes() {
    let (dir, data, mut cfg) = setup("dp-run-masked", 2, 0);
    cfg.trainer.target_mask = Some(TargetMaskSpec::PrefixLm {
        target_base: 12,
        always_score: Vec::new(),
    });
    run_world(args_for(&dir, &data, &cfg, "dp")).expect("masked world-2 run");
    let record = RunRecord::load(&dir.join("dp")).expect("run.json");
    assert_eq!(record.status, RunStatus::Completed, "{record:?}");
    assert!(record.last_loss.is_some_and(f64::is_finite), "{record:?}");
    let _ = std::fs::remove_dir_all(dir);
}

/// A GPU list must name one GPU per rank of the group.
#[test]
fn a_gpu_list_unlike_the_group_is_an_error() {
    let (dir, data, cfg) = setup("dp-run-gpu-count", 2, 0);
    let mut args = args_for(&dir, &data, &cfg, "dp");
    args.gpus = Some("0".to_string());
    let err = format!("{:#}", run_world(args).expect_err("1 GPU for 2 ranks"));
    assert!(
        err.contains("names 1 GPUs, and the rank group has 2 ranks"),
        "{err}"
    );
    let _ = std::fs::remove_dir_all(dir);
}
