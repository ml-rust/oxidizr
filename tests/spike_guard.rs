//! The loss-spike guard: skips, warnings, exact resume with skips, and data-parallel agreement.
//!
//! A loss probe scripts every step loss: it alternates 1.0 and 1.1, and the
//! spike steps read 1000. The guard's decisions then follow the script alone,
//! while the model trains on real gradients.

mod common;

use boostr::runtime::{Communicator, LocalCommunicator};
use boostr::{CpuDevice, CpuRuntime};
use common::{
    bits_by_name, build_model, data_loader, scratch_dir, tiny_config, tokens, train_parts, trainer,
    TrainParts,
};
use oxidizr::checkpoint_set::{plan_optimizer_resume, CheckpointLabel};
use oxidizr::config::{CheckpointMeta, LitConfig};
use oxidizr::data::{DataPosition, LitDataLoader, RowOrder};
use oxidizr::dp::AbortHook;
use oxidizr::fit::JsonSink;
use oxidizr::metrics::{
    read_spike_guard_state, GpuMemory, LossProbe, SpikeAction, SpikeGuardConfig, SpikeGuardState,
};
use oxidizr::model::{load_checkpoint_into_model, LitGPT};
use oxidizr::trainer::LitTrainer;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

const STEPS: usize = 8;
const HALF: usize = STEPS / 2;
/// 0-based step indices whose scripted loss is a spike: one before the save, one after.
const SPIKES: [usize; 2] = [2, 6];

type Bits = BTreeMap<String, Vec<u32>>;

fn probe() -> LossProbe {
    Arc::new(|index: usize, _loss: f64| {
        if SPIKES.contains(&index) {
            1000.0
        } else {
            1.0 + 0.1 * (index % 2) as f64
        }
    })
}

fn guard(action: SpikeAction) -> SpikeGuardConfig {
    SpikeGuardConfig {
        window: 4,
        z_threshold: 6.0,
        min_steps: 2,
        action,
    }
}

fn config(dir: &Path, action: Option<SpikeAction>) -> LitConfig {
    let mut cfg = tiny_config(dir, STEPS);
    cfg.trainer.spike_guard = action.map(guard);
    cfg
}

/// What a run reached.
struct Run {
    bits: Bits,
    timestep: u64,
    state: Option<SpikeGuardState>,
}

/// Trains `steps` steps from the stream start with the probe, then calls `save`.
fn train_fresh(
    cfg: &LitConfig,
    steps: usize,
    save: impl FnOnce(&LitTrainer<'_, CpuRuntime>, &LitGPT<CpuRuntime>),
) -> Run {
    let model = build_model(cfg);
    let mut parts = train_parts(&model);
    let mut loader = data_loader(cfg);
    let layout = loader.row_layout().expect("row layout");
    let (timestep, state) = {
        let mut t =
            trainer(cfg, &mut parts, layout, 0, DataPosition::default()).with_loss_probe(probe());
        for _ in 0..steps {
            t.training_step(&model, &mut loader).expect("training step");
        }
        save(&t, &model);
        (t.optimizer_timestep(), t.spike_guard_state())
    };
    Run {
        bits: bits_by_name(&parts),
        timestep,
        state,
    }
}

/// Resumes from the step-`HALF` set in `dir` as `commands::train` does, then trains to `STEPS`.
fn resume(cfg: &LitConfig, dir: &Path) -> Run {
    let weights = dir
        .join(CheckpointLabel::Step(HALF).weights_file_name(&cfg.model.checkpoint_prefix()))
        .to_string_lossy()
        .into_owned();
    let meta = dir
        .join(CheckpointLabel::Step(HALF).meta_file_name())
        .to_string_lossy()
        .into_owned();
    let mut model = build_model(cfg);
    load_checkpoint_into_model(&mut model, &weights).expect("load weights");
    let plan = plan_optimizer_resume(Some(&weights), Some(&meta), HALF).expect("plan");
    let position = CheckpointMeta::load(&meta)
        .expect("meta")
        .data_position
        .expect("data position");
    let saved = read_spike_guard_state(Some(&meta), HALF).expect("spike guard state");
    let mut parts = train_parts(&model);
    let mut loader = data_loader(cfg);
    let layout = loader.row_layout().expect("row layout");
    loader.seek(position).expect("seek");
    let (timestep, state) = {
        let mut t = trainer(cfg, &mut parts, layout, HALF, position).with_loss_probe(probe());
        t.apply_optimizer_resume(&plan).expect("optimizer resume");
        t.restore_spike_guard(saved).expect("restore guard");
        for _ in HALF..STEPS {
            t.training_step(&model, &mut loader).expect("training step");
        }
        (t.optimizer_timestep(), t.spike_guard_state())
    };
    Run {
        bits: bits_by_name(&parts),
        timestep,
        state,
    }
}

fn state_bits(state: Option<SpikeGuardState>) -> Option<(u64, u64, u64, u64, u64)> {
    state.map(|s| {
        (
            s.mean.to_bits(),
            s.variance.to_bits(),
            s.observed,
            s.detected,
            s.skipped,
        )
    })
}

#[test]
fn a_resume_with_skipped_spikes_is_bit_identical() {
    let dir = scratch_dir("spike-resume");
    let cfg = config(&dir, Some(SpikeAction::Skip));
    let reference = train_fresh(&cfg, STEPS, |_, _| {});
    let again = train_fresh(&cfg, STEPS, |_, _| {});
    assert_eq!(
        reference.bits, again.bits,
        "the scripted run is not deterministic"
    );
    assert_eq!(reference.timestep, (STEPS - SPIKES.len()) as u64);
    let end = reference.state.expect("a guard");
    assert_eq!((end.detected, end.skipped), (2, 2));
    assert_eq!(end.observed, (STEPS - SPIKES.len()) as u64);

    train_fresh(&cfg, HALF, |t, model| {
        t.save_checkpoint(model, HALF).expect("save step set");
    });
    let meta = CheckpointMeta::load(
        &dir.join(CheckpointLabel::Step(HALF).meta_file_name())
            .to_string_lossy(),
    )
    .expect("meta");
    let saved = meta.spike_guard.expect("the meta records the guard");
    assert_eq!((saved.observed, saved.detected, saved.skipped), (3, 1, 1));
    // The data position counts the skipped step's batches too.
    let layout = data_loader(&cfg).row_layout().expect("row layout");
    assert_eq!(
        meta.data_position,
        Some(layout.position_after_steps(HALF, cfg.trainer.gradient_accumulation))
    );

    let resumed = resume(&cfg, &dir);
    assert_eq!(resumed.timestep, reference.timestep);
    assert_eq!(state_bits(resumed.state), state_bits(reference.state));
    assert_eq!(resumed.bits, reference.bits, "the resumed tensors differ");

    // Without the guard the spike steps train: the skips are what the comparison sees.
    let unguarded = train_fresh(&config(&dir, None), STEPS, |_, _| {});
    assert_eq!(unguarded.timestep, STEPS as u64);
    assert_ne!(unguarded.bits, reference.bits);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn warn_mode_counts_spikes_and_trains_like_no_guard() {
    let dir = scratch_dir("spike-warn");
    let warned = train_fresh(&config(&dir, Some(SpikeAction::Warn)), STEPS, |_, _| {});
    let unguarded = train_fresh(&config(&dir, None), STEPS, |_, _| {});
    assert_eq!(warned.bits, unguarded.bits);
    assert_eq!(warned.timestep, STEPS as u64);
    let state = warned.state.expect("a guard");
    assert_eq!((state.detected, state.skipped), (2, 0));
    assert_eq!(unguarded.state, None);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_run_without_the_guard_writes_no_guard_keys() {
    let dir = scratch_dir("spike-off");
    let cfg = config(&dir, None);
    train_fresh(&cfg, 2, |t, model| {
        t.save_checkpoint(model, 2).expect("save");
    });
    let text =
        std::fs::read_to_string(dir.join(CheckpointLabel::Step(2).meta_file_name())).expect("meta");
    assert!(!text.contains("spike_guard"), "{text}");
    let yaml = serde_yaml::to_string(&cfg).expect("yaml");
    assert!(
        !yaml.contains("spike_guard") && !yaml.contains("metrics"),
        "{yaml}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn headless_output_writes_a_spike_line_for_each_flagged_step() {
    let dir = scratch_dir("spike-json");
    let mut cfg = config(&dir.join("ckpt"), Some(SpikeAction::Skip));
    cfg.trainer.log_interval = 100;
    let model = build_model(&cfg);
    let mut parts = train_parts(&model);
    let mut loader = data_loader(&cfg);
    let layout = loader.row_layout().expect("row layout");
    let (mut out, mut err) = (Vec::new(), Vec::new());
    {
        let no_gpu: fn() -> Option<GpuMemory> = || None;
        let mut sink = JsonSink::new(&mut out, &mut err, no_gpu);
        let mut t =
            trainer(&cfg, &mut parts, layout, 0, DataPosition::default()).with_loss_probe(probe());
        t.fit_with_sink(&model, &mut loader, &mut sink)
            .expect("fit");
    }
    let stdout = String::from_utf8(out).expect("utf-8");
    let spikes: Vec<serde_json::Value> = stdout
        .lines()
        .filter(|l| l.starts_with(r#"{"spike""#))
        .map(|l| serde_json::from_str(l).expect("a JSON spike line"))
        .collect();
    let steps: Vec<u64> = spikes
        .iter()
        .map(|v| v["spike"]["step"].as_u64().expect("step"))
        .collect();
    let want: Vec<u64> = SPIKES.iter().map(|&i| i as u64 + 1).collect();
    assert_eq!(steps, want, "{stdout}");
    assert!(spikes.iter().all(|v| v["spike"]["action"] == "skip"));
    assert!(err.is_empty(), "{}", String::from_utf8_lossy(&err));
    let _ = std::fs::remove_dir_all(dir);
}

const WORLD: usize = 2;

/// Every rank's spike decisions match, so the replicas stay bit-equal.
#[test]
fn every_rank_skips_the_same_steps() {
    let dir = scratch_dir("spike-dp");
    let mut cfg = config(&dir, Some(SpikeAction::Skip));
    cfg.trainer.gradient_accumulation = 1;
    let data = tokens(&cfg, 4000);
    let comms: Vec<Arc<LocalCommunicator>> =
        LocalCommunicator::with_timeout(WORLD, Duration::from_secs(120))
            .into_iter()
            .map(Arc::new)
            .collect();
    let runs: Vec<(Bits, Vec<u64>, u64, Option<SpikeGuardState>)> = std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .iter()
            .map(|local| {
                let (cfg, data) = (&cfg, &data);
                s.spawn(move || {
                    let _guard = local.abort_on_panic();
                    let peer = Arc::clone(local);
                    let abort: AbortHook = Arc::new(move || peer.abort());
                    let comm: Arc<dyn Communicator> = Arc::clone(local) as Arc<dyn Communicator>;
                    rank_run(cfg, data, local.rank(), comm, abort)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank thread"))
            .collect()
    });
    let (bits, losses, timestep, state) = &runs[0];
    for (rank, run) in runs.iter().enumerate().skip(1) {
        assert_eq!(&run.0, bits, "rank {rank} tensors differ");
        assert_eq!(&run.1, losses, "rank {rank} losses differ");
        assert_eq!(run.2, *timestep, "rank {rank} timestep");
        assert_eq!(
            state_bits(run.3),
            state_bits(*state),
            "rank {rank} guard state"
        );
    }
    assert_eq!(*timestep, (STEPS - SPIKES.len()) as u64);
    assert_eq!(state.map(|s| s.skipped), Some(SPIKES.len() as u64));
    let _ = std::fs::remove_dir_all(dir);
}

/// One rank of [`every_rank_skips_the_same_steps`]: its tensors, loss bits, timestep and guard.
fn rank_run(
    cfg: &LitConfig,
    data: &[u32],
    rank: usize,
    comm: Arc<dyn Communicator>,
    abort: AbortHook,
) -> (Bits, Vec<u64>, u64, Option<SpikeGuardState>) {
    let model = build_model(cfg);
    let mut parts: TrainParts = train_parts(&model);
    let mut loader = LitDataLoader::<CpuRuntime>::new(
        data.to_vec(),
        cfg.trainer.batch_size,
        cfg.model.max_seq_len,
        CpuDevice::new(),
    )
    .with_row_order(RowOrder::Shuffled { seed: 7 })
    .with_rank(rank, WORLD)
    .expect("rank loader");
    let layout = loader.row_layout().expect("row layout");
    let mut losses = Vec::new();
    let (timestep, state) = {
        let mut t = trainer(cfg, &mut parts, layout, 0, DataPosition::default())
            .with_communicator(comm)
            .expect("communicator")
            .with_abort_hook(abort)
            .with_loss_probe(probe());
        t.sync_initial_parameters(&model).expect("initial sync");
        for _ in 0..STEPS {
            let (loss, _) = t.training_step(&model, &mut loader).expect("training step");
            losses.push(loss.to_bits());
        }
        (t.optimizer_timestep(), t.spike_guard_state())
    };
    (bits_by_name(&parts), losses, timestep, state)
}
