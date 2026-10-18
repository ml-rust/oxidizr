//! Data-parallel steps over an injected communicator match one rank on the merged batch.
//!
//! Each rank runs on its own thread, with its own CPU client, trainer and rank
//! loader. The reference is one trainer whose batch holds every rank's batch,
//! on the same data, seed and config. Ranks must stay bit-equal to each other
//! and close to the reference.

mod common;

use boostr::optimizer::GradScalerState;
use boostr::runtime::{Communicator, LocalCommunicator, NoOpCommunicator};
use boostr::{CpuDevice, CpuRuntime};
use common::{
    bits_by_name, build_model, scratch_dir, tiny_config, tokens, train_parts, trainer, TrainParts,
};
use oxidizr::config::{LitConfig, LoraConfig};
use oxidizr::data::{DataPosition, LitDataLoader, RowOrder, TargetMask};
use oxidizr::dp::AbortHook;
use oxidizr::fit::NullSink;
use oxidizr::model::LitGPT;
use oxidizr::trainer::LitTrainer;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Barrier};
use std::time::Duration;

const WORLD: usize = 2;
/// First scored target id of the masked data. The tiny vocab holds 24 ids.
const SCORED_BASE: u32 = 12;
/// Group timeout of the equivalence runs: long enough never to fire.
const TIMEOUT: Duration = Duration::from_secs(120);

type Bits = BTreeMap<String, Vec<u32>>;

/// The tokens a run reads, their row order, and the optional prefix-LM mask.
struct Data {
    tokens: Vec<u32>,
    order: RowOrder,
    /// First scored target id. `None` scores every position.
    target_base: Option<u32>,
}

impl Data {
    /// The shuffled, unmasked token stream of `common::data_loader`.
    fn shuffled(cfg: &LitConfig) -> Self {
        Self {
            tokens: tokens(cfg, 4000),
            order: RowOrder::Shuffled { seed: 7 },
            target_base: None,
        }
    }

    /// Stream-order rows whose scored ids follow `scored(slot, pos)`.
    ///
    /// `slot` is the row's slot in its global batch of `batch_size * WORLD` rows.
    /// Rank `k` reads slots `[k * batch_size, (k + 1) * batch_size)`.
    /// `pos` is the token's position in its row.
    fn masked(cfg: &LitConfig, rows: usize, scored: impl Fn(usize, usize) -> bool) -> Self {
        let width = cfg.model.max_seq_len + 1;
        let global = cfg.trainer.batch_size * WORLD;
        let tokens = (0..rows * width)
            .map(|i| {
                let low = ((i * 7 + i / 5) % SCORED_BASE as usize) as u32;
                if scored((i / width) % global, i % width) {
                    low + SCORED_BASE
                } else {
                    low
                }
            })
            .collect();
        Self {
            tokens,
            order: RowOrder::Stream,
            target_base: Some(SCORED_BASE),
        }
    }

    fn loader(&self, cfg: &LitConfig, batch_size: usize) -> LitDataLoader<CpuRuntime> {
        let loader = LitDataLoader::<CpuRuntime>::new(
            self.tokens.clone(),
            batch_size,
            cfg.model.max_seq_len,
            CpuDevice::new(),
        )
        .with_row_order(self.order);
        match self.target_base {
            Some(target_base) => loader.with_target_mask(TargetMask::PrefixLm {
                target_base,
                always_score: Vec::new(),
            }),
            None => loader,
        }
    }

    /// Rank `rank` of `WORLD`, at the rank batch size of `cfg`.
    fn rank_loader(&self, cfg: &LitConfig, rank: usize) -> LitDataLoader<CpuRuntime> {
        self.loader(cfg, cfg.trainer.batch_size)
            .with_rank(rank, WORLD)
            .expect("valid rank")
    }
}

/// What one run reached.
#[derive(Debug)]
struct Run {
    /// Trainable tensors before the first step.
    initial: Bits,
    losses: Vec<f64>,
    norms: Vec<f64>,
    bits: Bits,
    timestep: u64,
    loss_scale: Option<GradScalerState>,
}

/// Run `steps` steps on `t` and record what it reached, except the tensors.
fn train(
    t: &mut LitTrainer<'_, CpuRuntime>,
    model: &LitGPT<CpuRuntime>,
    loader: &mut LitDataLoader<CpuRuntime>,
    steps: usize,
) -> (Vec<f64>, Vec<f64>, u64, Option<GradScalerState>) {
    let mut losses = Vec::new();
    let mut norms = Vec::new();
    for _ in 0..steps {
        let (loss, norm) = t.training_step(model, loader).expect("training step");
        losses.push(loss);
        norms.push(norm);
    }
    (losses, norms, t.optimizer_timestep(), t.loss_scale_state())
}

/// One rank whose batch holds the batches of all `WORLD` ranks of `cfg`.
fn single_run(cfg: &LitConfig, data: &Data, steps: usize) -> Run {
    let mut cfg = cfg.clone();
    cfg.trainer.batch_size *= WORLD;
    let model = build_model(&cfg);
    let mut parts = train_parts(&model);
    let initial = bits_by_name(&parts);
    let mut loader = data.loader(&cfg, cfg.trainer.batch_size);
    let layout = loader.row_layout().expect("row layout");
    let (losses, norms, timestep, loss_scale) = {
        let mut t = trainer(&cfg, &mut parts, layout, 0, DataPosition::default());
        train(&mut t, &model, &mut loader, steps)
    };
    Run {
        initial,
        losses,
        norms,
        bits: bits_by_name(&parts),
        timestep,
        loss_scale,
    }
}

/// Run `body(rank, comm, abort)` on one thread per rank of a fresh group.
///
/// The group outlives every thread, so no rank's exit aborts its peers.
fn on_ranks<T: Send>(
    timeout: Duration,
    body: impl Fn(usize, Arc<dyn Communicator>, AbortHook) -> T + Sync,
) -> Vec<T> {
    let comms: Vec<Arc<LocalCommunicator>> = LocalCommunicator::with_timeout(WORLD, timeout)
        .into_iter()
        .map(Arc::new)
        .collect();
    std::thread::scope(|s| {
        let body = &body;
        let handles: Vec<_> = comms
            .iter()
            .map(|local| {
                s.spawn(move || {
                    let _guard = local.abort_on_panic();
                    let peer = Arc::clone(local);
                    let abort: AbortHook = Arc::new(move || peer.abort());
                    let comm: Arc<dyn Communicator> = Arc::clone(local) as Arc<dyn Communicator>;
                    body(local.rank(), comm, abort)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank thread"))
            .collect()
    })
}

/// A trainer for rank `rank` over `parts`, joined to `comm`.
fn rank_trainer<'a>(
    cfg: &LitConfig,
    parts: &'a mut TrainParts,
    loader: &LitDataLoader<CpuRuntime>,
    comm: Arc<dyn Communicator>,
    abort: AbortHook,
) -> LitTrainer<'a, CpuRuntime> {
    let layout = loader.row_layout().expect("row layout");
    trainer(cfg, parts, layout, 0, DataPosition::default())
        .with_communicator(comm)
        .expect("communicator matches the layout")
        .with_abort_hook(abort)
}

/// `WORLD` ranks, each at the rank batch size of `cfg`, for `steps` steps.
fn dp_run(cfg: &LitConfig, data: &Data, steps: usize) -> Vec<Run> {
    on_ranks(TIMEOUT, |rank, comm, abort| {
        let model = build_model(cfg);
        let mut parts = train_parts(&model);
        let initial = bits_by_name(&parts);
        let mut loader = data.rank_loader(cfg, rank);
        let (losses, norms, timestep, loss_scale) = {
            let mut t = rank_trainer(cfg, &mut parts, &loader, comm, abort);
            t.sync_initial_parameters(&model).expect("initial sync");
            train(&mut t, &model, &mut loader, steps)
        };
        Run {
            initial,
            losses,
            norms,
            bits: bits_by_name(&parts),
            timestep,
            loss_scale,
        }
    })
}

/// Every rank holds the same tensors, losses, norms and optimizer state, bit for bit.
fn assert_ranks_bit_equal(runs: &[Run]) {
    let first = &runs[0];
    for (rank, run) in runs.iter().enumerate().skip(1) {
        assert_eq!(
            run.bits, first.bits,
            "rank {rank} tensors differ from rank 0"
        );
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&run.losses), bits(&first.losses), "rank {rank} loss");
        assert_eq!(bits(&run.norms), bits(&first.norms), "rank {rank} norm");
        assert_eq!(run.timestep, first.timestep, "rank {rank} timestep");
        assert_eq!(run.loss_scale, first.loss_scale, "rank {rank} loss scale");
    }
}

/// The data-parallel run matches the single run: losses within 1e-6 relative,
/// every element within 1e-6 absolute plus 1e-5 relative.
fn assert_close_to_single(dp: &Run, single: &Run) {
    assert_eq!(dp.losses.len(), single.losses.len());
    for (step, (got, want)) in dp.losses.iter().zip(&single.losses).enumerate() {
        let tolerance = 1e-6 * want.abs().max(f64::MIN_POSITIVE);
        assert!(
            (got - want).abs() <= tolerance,
            "step {step}: loss {got} != single-rank {want}"
        );
    }
    assert_eq!(dp.timestep, single.timestep, "optimizer timestep");
    assert_eq!(dp.bits.len(), single.bits.len());
    for (name, want) in &single.bits {
        let got = &dp.bits[name];
        assert_eq!(got.len(), want.len(), "{name} length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let (g, w) = (f32::from_bits(*g), f32::from_bits(*w));
            assert!(
                (g - w).abs() <= 1e-6 + 1e-5 * w.abs(),
                "{name}[{i}]: data-parallel {g} != single-rank {w}"
            );
        }
    }
    assert_ne!(
        single.bits, single.initial,
        "the single run trained nothing"
    );
}

/// Both runs for `steps` steps, checked against each other.
fn check_equivalence(cfg: &LitConfig, data: &Data, steps: usize) {
    let runs = dp_run(cfg, data, steps);
    assert_ranks_bit_equal(&runs);
    assert_close_to_single(&runs[0], &single_run(cfg, data, steps));
}

/// `tiny_config` with rank batch 2 and `accum` micro-batches per step.
fn dp_config(tag: &str, steps: usize, accum: usize) -> (LitConfig, std::path::PathBuf) {
    let dir = scratch_dir(tag);
    let mut cfg = tiny_config(&dir, steps);
    cfg.trainer.batch_size = 2;
    cfg.trainer.gradient_accumulation = accum;
    (cfg, dir)
}

#[test]
fn unmasked_ranks_match_one_rank_on_the_merged_batch() {
    let (cfg, dir) = dp_config("dp-unmasked", 3, 1);
    let data = Data::shuffled(&cfg);
    for steps in [1, 3] {
        check_equivalence(&cfg, &data, steps);
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn gradient_accumulation_matches_the_merged_batch() {
    // Rank batch 2 x 2 ranks x 2 micro-batches. The single run reads batch 4 x 2.
    let (cfg, dir) = dp_config("dp-accum", 3, 2);
    check_equivalence(&cfg, &Data::shuffled(&cfg), 3);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn uneven_masked_ranks_match_the_merged_batch() {
    let (cfg, dir) = dp_config("dp-masked", 3, 1);
    let rank_batch = cfg.trainer.batch_size;
    // Rank 0 rows score 4 targets each, rank 1 rows score 2.
    let data = Data::masked(&cfg, 200, |slot, pos| {
        if slot < rank_batch {
            pos % 2 == 0
        } else {
            pos % 4 == 1
        }
    });
    let scored: Vec<usize> = (0..WORLD)
        .map(|rank| {
            let mut loader = data.rank_loader(&cfg, rank);
            loader.next_batch().expect("rank batch").scored
        })
        .collect();
    assert!(
        scored[0] > 0 && scored[1] > 0 && scored[0] != scored[1],
        "{scored:?}"
    );
    for steps in [1, 3] {
        check_equivalence(&cfg, &data, steps);
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// MoE routing is per token. With `load_balance_alpha: 0` the auxiliary loss
/// adds no gradient, so the merged batch is still the reference. Top-1 routing
/// leaves some experts without a gradient on one rank.
#[test]
fn moe_without_aux_weight_matches_the_merged_batch() {
    let (mut cfg, dir) = dp_config("dp-moe", 3, 1);
    cfg.model.num_experts = Some(2);
    cfg.model.experts_per_tok = Some(1);
    cfg.trainer.load_balance_alpha = 0.0;
    cfg.trainer.z_loss_alpha = 0.0;
    let data = Data::shuffled(&cfg);
    for steps in [1, 3] {
        check_equivalence(&cfg, &data, steps);
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// Rank 1 scores no target, so it has no gradient. Rank 0's loss scale puts
/// its backward seed past f16's maximum. The summed gradient overflows on
/// both ranks, so both skip and both halve the scale.
#[cfg(feature = "f16")]
#[test]
fn an_overflow_on_one_rank_skips_the_step_on_every_rank() {
    use oxidizr::config::DTypePrecision;
    use oxidizr::loss_scale::LossScaleConfig;

    let (mut cfg, dir) = dp_config("dp-f16-overflow", 1, 1);
    cfg.model.dtype = DTypePrecision::F16;
    cfg.trainer.loss_scale = Some(LossScaleConfig {
        initial: 1048576.0,
        ..LossScaleConfig::default()
    });
    let rank_batch = cfg.trainer.batch_size;
    let data = Data::masked(&cfg, 200, |slot, pos| slot < rank_batch && pos % 2 == 0);
    let runs = dp_run(&cfg, &data, 1);
    assert_ranks_bit_equal(&runs);
    for (rank, run) in runs.iter().enumerate() {
        assert!(
            !run.norms[0].is_finite(),
            "rank {rank}: norm {}",
            run.norms[0]
        );
        assert!(
            run.losses[0].is_finite(),
            "rank {rank}: loss {}",
            run.losses[0]
        );
        assert_eq!(run.timestep, 0, "rank {rank} stepped the optimizer");
        assert_eq!(run.bits, run.initial, "rank {rank} changed a parameter");
        assert_eq!(
            run.loss_scale,
            Some(GradScalerState {
                scale: 524288.0,
                consecutive_ok: 0
            }),
            "rank {rank}"
        );
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn world_one_with_an_injected_noop_is_bit_identical_to_the_default() {
    let dir = scratch_dir("dp-world-one");
    let cfg = tiny_config(&dir, 3);
    let data = Data::shuffled(&cfg);
    let run = |inject: bool| {
        let model = build_model(&cfg);
        let mut parts = train_parts(&model);
        let mut loader = data.loader(&cfg, cfg.trainer.batch_size);
        let layout = loader.row_layout().expect("row layout");
        let (losses, norms, _, _) = {
            let mut t = trainer(&cfg, &mut parts, layout, 0, DataPosition::default());
            if inject {
                t = t
                    .with_communicator(Arc::new(NoOpCommunicator))
                    .expect("world 1");
            }
            assert_eq!((t.rank(), t.world_size()), (0, 1));
            t.sync_initial_parameters(&model).expect("no-op sync");
            train(&mut t, &model, &mut loader, 3)
        };
        let bits = |v: Vec<f64>| v.into_iter().map(f64::to_bits).collect::<Vec<_>>();
        (bits(losses), bits(norms), bits_by_name(&parts))
    };
    assert_eq!(run(true), run(false));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_communicator_world_unlike_the_layout_is_an_error() {
    let dir = scratch_dir("dp-world-mismatch");
    let cfg = tiny_config(&dir, 1);
    let model = build_model(&cfg);
    let mut parts = train_parts(&model);
    // A world-1 layout cannot take a rank of a two-rank group.
    let layout = Data::shuffled(&cfg)
        .loader(&cfg, cfg.trainer.batch_size)
        .row_layout()
        .expect("row layout");
    let comms = LocalCommunicator::group(WORLD);
    let comm: Arc<dyn Communicator> = Arc::new(comms.into_iter().next().expect("rank 0"));
    let Err(err) =
        trainer(&cfg, &mut parts, layout, 0, DataPosition::default()).with_communicator(comm)
    else {
        panic!("a world-2 communicator must not join a world-1 layout");
    };
    assert!(err.to_string().contains("world size 2"), "{err}");
    let _ = std::fs::remove_dir_all(dir);
}

/// Every tensor of `model`, frozen ones included, by name.
fn model_bits(model: &LitGPT<CpuRuntime>, parts: &TrainParts) -> Bits {
    model
        .named_parameters()
        .expect("named parameters")
        .into_iter()
        .map(|(name, id, var)| {
            let tensor = parts.params.get(&id).unwrap_or(var.tensor());
            let values = tensor.contiguous().expect("contiguous").to_vec::<f32>();
            (name, values.iter().map(|v| v.to_bits()).collect())
        })
        .collect()
}

/// Ranks built from different seeds hold rank 0's tensors after the sync,
/// frozen LoRA base weights included, and stay bit-equal after a step.
#[test]
fn the_initial_broadcast_gives_every_rank_rank_zeros_tensors() {
    let dir = scratch_dir("dp-initial-sync");
    let mut cfg = tiny_config(&dir, 1);
    cfg.trainer.gradient_accumulation = 1;
    cfg.model.lora = Some(LoraConfig {
        rank: 2,
        alpha: 4.0,
        target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        freeze_base: true,
        train_modules: Vec::new(),
    });
    let data = Data::shuffled(&cfg);
    let results = on_ranks(TIMEOUT, |rank, comm, abort| {
        let mut rank_cfg = cfg.clone();
        rank_cfg.trainer.seed = Some(100 + rank as u64);
        let model = build_model(&rank_cfg);
        let mut parts = train_parts(&model);
        let before = model_bits(&model, &parts);
        let mut loader = data.rank_loader(&rank_cfg, rank);
        {
            let comm = Arc::clone(&comm);
            let mut t = rank_trainer(&rank_cfg, &mut parts, &loader, comm, Arc::clone(&abort));
            t.sync_initial_parameters(&model).expect("initial sync");
        }
        let synced = model_bits(&model, &parts);
        {
            let mut t = rank_trainer(&rank_cfg, &mut parts, &loader, comm, abort);
            t.training_step(&model, &mut loader).expect("training step");
        }
        (before, synced, bits_by_name(&parts))
    });
    let (rank0_before, _, rank0_after) = &results[0];
    let (rank1_before, rank1_synced, rank1_after) = &results[1];
    assert_ne!(rank0_before, rank1_before, "the seeds built equal tensors");
    assert!(
        rank0_before.keys().any(|name| name.contains("lora")),
        "the model holds no adapter"
    );
    assert_eq!(
        rank1_synced, rank0_before,
        "rank 1 does not hold rank 0's tensors"
    );
    assert_eq!(
        rank1_after, rank0_after,
        "the replicas diverged after one step"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// Rank 1 fails before any collective. Its abort hook wakes rank 0, which
/// waits in the step-count sum, so rank 0 returns an error instead of hanging.
#[test]
fn a_failed_rank_aborts_its_peers() {
    let (cfg, dir) = dp_config("dp-failed-rank", 1, 1);
    let data = Data::shuffled(&cfg);
    let results = on_ranks(TIMEOUT, |rank, comm, abort| {
        let model = build_model(&cfg);
        let mut parts = train_parts(&model);
        let mut loader = data.rank_loader(&cfg, rank);
        let mut t = rank_trainer(&cfg, &mut parts, &loader, comm, abort);
        t.sync_initial_parameters(&model).expect("initial sync");
        if rank == 1 {
            // Three rows cannot fill one global batch of four.
            let short = Data {
                tokens: tokens(&cfg, 30),
                ..Data::shuffled(&cfg)
            };
            loader = short.rank_loader(&cfg, rank);
        }
        t.training_step(&model, &mut loader)
            .err()
            .map(|e| e.to_string())
    });
    let rank0 = results[0].as_deref().expect("rank 0 must fail");
    let rank1 = results[1].as_deref().expect("rank 1 must fail");
    assert!(rank1.contains("global batch"), "{rank1}");
    assert!(
        rank0.contains("aborted"),
        "rank 0 did not see the abort: {rank0}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// A collective that times out is the step's error. The group timeout is
/// short, and rank 1 never joins the step.
#[test]
fn a_collective_timeout_is_the_step_error() {
    let (cfg, dir) = dp_config("dp-timeout", 1, 1);
    let data = Data::shuffled(&cfg);
    let done = Barrier::new(WORLD);
    let results = on_ranks(Duration::from_secs(1), |rank, comm, abort| {
        if rank == 1 {
            // Hold the group open without joining, until rank 0 has its error.
            done.wait();
            return None;
        }
        let model = build_model(&cfg);
        let mut parts = train_parts(&model);
        let mut loader = data.rank_loader(&cfg, rank);
        let result = {
            let mut t = rank_trainer(&cfg, &mut parts, &loader, comm, abort);
            t.training_step(&model, &mut loader)
        };
        done.wait();
        result.err().map(|e| e.to_string())
    });
    let rank0 = results[0].as_deref().expect("rank 0 must fail");
    assert!(rank0.contains("timed out"), "{rank0}");
    let _ = std::fs::remove_dir_all(dir);
}

/// Rank 0 masks and rank 1 does not. Both join the same step-count sum, so
/// both return the mask error before any forward pass, and no rank hangs.
#[test]
fn mixed_mask_presence_fails_on_every_rank() {
    let (cfg, dir) = dp_config("dp-mismatch", 1, 1);
    let masked = Data {
        target_base: Some(SCORED_BASE),
        ..Data::shuffled(&cfg)
    };
    let unmasked = Data::shuffled(&cfg);
    let results = on_ranks(TIMEOUT, |rank, comm, abort| {
        let model = build_model(&cfg);
        let mut parts = train_parts(&model);
        let data = if rank == 0 { &masked } else { &unmasked };
        let mut loader = data.rank_loader(&cfg, rank);
        let mut t = rank_trainer(&cfg, &mut parts, &loader, comm, abort);
        t.training_step(&model, &mut loader)
            .err()
            .map(|e| e.to_string())
    });
    for (rank, result) in results.iter().enumerate() {
        let text = result.as_deref().expect("every rank must fail");
        assert!(
            text.contains("1 of 2 data-parallel ranks"),
            "rank {rank}: {text}"
        );
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// Entries of `dir`, or none when it does not exist.
fn entries(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .map(|it| {
            it.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default()
}

/// The fit loop writes config, steps and checkpoints on rank 0 only. Every
/// rank ends with the same tensors after the final barrier.
#[test]
fn fit_writes_files_on_rank_zero_only() {
    let (base, root) = dp_config("dp-fit", 2, 1);
    let data = Data::shuffled(&base);
    let dirs: Vec<_> = (0..WORLD).map(|r| root.join(format!("rank{r}"))).collect();
    let results = on_ranks(TIMEOUT, |rank, comm, abort| {
        let mut cfg = base.clone();
        cfg.trainer.checkpoint_dir = dirs[rank].to_string_lossy().into_owned();
        cfg.trainer.save_interval = 1;
        let model = build_model(&cfg);
        let mut parts = train_parts(&model);
        let mut loader = data.rank_loader(&cfg, rank);
        {
            let mut t = rank_trainer(&cfg, &mut parts, &loader, comm, abort);
            t.fit_with_sink(&model, &mut loader, &mut NullSink)
                .expect("fit");
            assert_eq!(t.global_step(), 2, "rank {rank}");
        }
        bits_by_name(&parts)
    });
    assert_eq!(results[0], results[1], "the replicas diverged");
    let rank0 = entries(&dirs[0]);
    for name in ["steps.json", "config.json", "checkpoint_step_2.meta.json"] {
        assert!(
            rank0.iter().any(|e| e == name),
            "rank 0 lacks {name}: {rank0:?}"
        );
    }
    assert!(
        rank0.iter().any(|e| e.ends_with("_final.safetensors")),
        "rank 0 lacks the final weights: {rank0:?}"
    );
    assert_eq!(
        entries(&dirs[1]),
        Vec::<String>::new(),
        "rank 1 wrote files"
    );
    let _ = std::fs::remove_dir_all(root);
}
