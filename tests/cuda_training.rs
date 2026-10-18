//! CUDA training runs of a tiny Gated DeltaNet + attention hybrid, through the library path `commands::train`
//! takes: `LitGPT::new_seeded`, a safetensors load, `LitTrainer::training_step`.
//!
//! - A falling loss over three steps on a repeated row.
//! - Finite gradients on every step. The step's global gradient norm covers
//!   every trainable gradient, and a norm is finite only when each element is.
//! - Parameters after three steps against the same run on CPU.
//! - A bit-identical interrupt and resume on CUDA, when two uninterrupted CUDA
//!   runs already agree bit for bit.
//! - f16 with dynamic loss scaling, and bf16: overflow steps skip, losses stay finite.
//!
//! The CPU and CUDA runs start from one CPU-built weight file, because the
//! seeded draws of the two backends differ.
//!
//! # Tolerances
//!
//! - Parameters after three steps: `max |cuda - cpu| <= 1e-3 * max(max |cpu|, 1e-3)`
//!   per tensor. The floor stops a near-zero tensor from failing on rounding noise. Forward and backward sum in another order on CUDA. AdamW divides
//!   by the root of the second moment, which turns a gradient gap of a few
//!   ulps into a slightly larger step gap, and three steps add three of them.
//! - Step losses: `|cuda - cpu| <= 1e-3 * |cpu|`, for the same reasons.
//!
//! # Model
//!
//! The CUDA flash-attention kernel supports head dims 32, 64, 96, 128, 192 and
//! 256. The model uses hidden size 64 with 2 query heads and 1 KV head, so the
//! attention head dim is 32. GDN uses state size 32, one key head and two
//! value heads, which stays inside the fused kernel's supported sizes. The CPU
//! and CUDA runs of each case use this one config.
//!
//! Run with:
//!   cd oxidizr && cargo nextest run --features cuda --test cuda_training

#![cfg(feature = "cuda")]

mod common;

use boostr::{
    CpuClient, CpuDevice, CpuRuntime, CudaClient, CudaDevice, CudaRuntime, DType, Runtime,
};
use common::{
    build_model, build_model_on, scratch_dir, tiny_config, tiny_model_config, tokens,
    train_parts_on, trainer_on, values_by_name_on, TrainClient,
};
use oxidizr::checkpoint_set::{plan_optimizer_resume, OptimizerResume};
use oxidizr::config::{
    find_latest_checkpoint, CheckpointMeta, GdnSpec, LitConfig, OxidizrModelConfig, TargetDevice,
};
use oxidizr::data::{DataPosition, LitDataLoader, RowOrder};
use oxidizr::loss_scale::read_loss_scale_state;
use oxidizr::model::{load_checkpoint_into_model, save_checkpoint_from_params, LitGPT};
use oxidizr::schedule::LrSchedule;
use oxidizr::weight_decay::{inherited_weight_decay, resolve_weight_decay};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;
use std::sync::{Mutex, MutexGuard, OnceLock};

const STEPS: usize = 3;
/// The interrupted run saves after this many steps and resumes from there.
const RESUME_AT: usize = 1;
const PARAM_RTOL: f32 = 1e-3;
const LOSS_RTOL: f64 = 1e-3;

type Values = BTreeMap<String, Vec<f32>>;

static CUDA_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// The lock guard and the CUDA backend, or `None` without a CUDA device.
///
/// The guard serializes the cases: the device client is process-wide with
/// one stream.
fn cuda() -> Option<(MutexGuard<'static, ()>, CudaClient, CudaDevice)> {
    let guard = CUDA_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    if !boostr::runtime::cuda::is_cuda_available() {
        eprintln!("no CUDA device: skipping the CUDA training case");
        return None;
    }
    let device = CudaDevice::new(0);
    Some((guard, CudaRuntime::default_client(&device), device))
}

fn cpu() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    (CpuClient::new(device.clone()), device)
}

/// The rows a run trains on.
#[derive(Clone, Copy)]
enum Stream {
    /// The common token stream in a seeded shuffled row order.
    Shuffled,
    /// One row over and over, so every step scores the same batch.
    Repeated,
}

/// GDN layer 0 and attention layer 1, with attention head dim 32.
fn cuda_gdn_model_config() -> OxidizrModelConfig {
    OxidizrModelConfig {
        name: "gdn-cuda-test".to_string(),
        hidden_size: 64,
        num_heads: 2,
        kv_heads: Some(1),
        intermediate_size: Some(128),
        gdn_layers: Some(vec![0]),
        gdn: Some(GdnSpec {
            state_size: 32,
            key_heads: 1,
            value_heads: 2,
            conv_kernel: 4,
            chunk_size: 4,
            rms_eps: 1e-6,
        }),
        ..tiny_model_config(false)
    }
}

/// The GDN hybrid at a constant learning rate, `STEPS` steps.
fn gdn_config(dir: &Path) -> LitConfig {
    let mut cfg = tiny_config(dir, STEPS);
    cfg.model = cuda_gdn_model_config();
    cfg.trainer.learning_rate = 5e-3;
    cfg.trainer.warmup_steps = None;
    cfg.trainer.lr_schedule = LrSchedule::Constant;
    cfg.trainer.target_device = TargetDevice::Gpu;
    cfg
}

fn loader_on<R>(cfg: &LitConfig, stream: Stream, device: R::Device) -> LitDataLoader<R>
where
    R: Runtime<DType = DType>,
{
    let (batch, seq) = (cfg.trainer.batch_size, cfg.model.max_seq_len);
    match stream {
        Stream::Shuffled => LitDataLoader::<R>::new(tokens(cfg, 4000), batch, seq, device)
            .with_row_order(RowOrder::Shuffled { seed: 7 }),
        Stream::Repeated => {
            let row_len = seq + 1;
            let vocab = cfg.model.vocab_size as u32;
            let row: Vec<u32> = (0..row_len as u32).map(|i| (i * 5 + 1) % vocab).collect();
            let all = row.iter().cycle().take(row_len * 256).copied().collect();
            LitDataLoader::<R>::new(all, batch, seq, device)
        }
    }
}

/// Writes the CPU-built weights of `cfg` to `dir/init.safetensors`.
fn write_init(cfg: &LitConfig, dir: &Path) -> String {
    let model = build_model(cfg);
    let params: HashMap<_, _> = model
        .named_parameters()
        .expect("named parameters")
        .into_iter()
        .map(|(_, id, var)| (id, var.tensor().clone()))
        .collect();
    let path = dir.join("init.safetensors").to_string_lossy().into_owned();
    save_checkpoint_from_params(&path, &model, &params).expect("save the initial weights");
    path
}

/// The model of `cfg` on `backend`, holding the tensors of `weights` when given.
fn model_on<R>(
    cfg: &LitConfig,
    weights: Option<&str>,
    backend: &(R::Client, R::Device),
) -> LitGPT<R>
where
    R: Runtime<DType = DType>,
    R::Client: TrainClient<R>,
{
    let (client, device) = backend;
    let mut model = build_model_on::<R>(cfg, device, client);
    if let Some(path) = weights {
        load_checkpoint_into_model(&mut model, path).expect("load weights");
    }
    model
}

/// Losses, clipped gradient norms and final trainable values of one run.
struct Run {
    losses: Vec<f64>,
    norms: Vec<f64>,
    values: Values,
}

/// `steps` steps from the stream start, over the weights in `init` when
/// given. Saves the checkpoint set after step `save_at` when given.
fn fresh<R>(
    cfg: &LitConfig,
    init: Option<&str>,
    stream: Stream,
    backend: (R::Client, R::Device),
    steps: usize,
    save_at: Option<usize>,
) -> Run
where
    R: Runtime<DType = DType>,
    R::Client: TrainClient<R>,
{
    let model = model_on::<R>(cfg, init, &backend);
    let mut parts = train_parts_on(&model);
    let mut loader = loader_on::<R>(cfg, stream, backend.1.clone());
    let layout = loader.row_layout().expect("row layout");
    loader.seek(DataPosition::default()).expect("seek");
    let weight_decay = resolve_weight_decay(&cfg.trainer, None, true).expect("weight decay rule");
    let client = backend.0.clone();
    let (mut losses, mut norms) = (Vec::new(), Vec::new());
    {
        let start = (0, DataPosition::default());
        let mut t = trainer_on(cfg, &mut parts, layout, start, weight_decay, backend);
        for step in 1..=steps {
            let (loss, norm) = t.training_step(&model, &mut loader).expect("training step");
            losses.push(loss);
            norms.push(norm);
            if save_at == Some(step) {
                t.save_checkpoint(&model, step).expect("save the step set");
            }
        }
    }
    Run {
        losses,
        norms,
        values: values_by_name_on(&parts, &client),
    }
}

/// Resume from `weights` and `meta` as `commands::train` does, then train to `STEPS`.
fn resume<R>(
    cfg: &LitConfig,
    stream: Stream,
    backend: (R::Client, R::Device),
    weights: &str,
    meta: &str,
    start_step: usize,
) -> Run
where
    R: Runtime<DType = DType>,
    R::Client: TrainClient<R>,
{
    let model = model_on::<R>(cfg, Some(weights), &backend);
    let plan = plan_optimizer_resume(Some(weights), Some(meta), start_step).expect("plan");
    assert!(matches!(plan, OptimizerResume::Load(_)), "{plan:?}");
    let position = CheckpointMeta::load(meta)
        .expect("meta")
        .data_position
        .expect("data position");
    let inherited = inherited_weight_decay(Some(meta), start_step).expect("inherited rule");
    let weight_decay =
        resolve_weight_decay(&cfg.trainer, inherited.as_ref(), true).expect("weight decay rule");
    let saved_scale = read_loss_scale_state(Some(meta), start_step).expect("loss scale state");
    let mut parts = train_parts_on(&model);
    let mut loader = loader_on::<R>(cfg, stream, backend.1.clone());
    let layout = loader.row_layout().expect("row layout");
    loader.seek(position).expect("seek");
    let client = backend.0.clone();
    let (mut losses, mut norms) = (Vec::new(), Vec::new());
    {
        let start = (start_step, position);
        let mut t = trainer_on(cfg, &mut parts, layout, start, weight_decay, backend);
        t.apply_optimizer_resume(&plan)
            .expect("apply the optimizer plan");
        t.restore_loss_scale(saved_scale)
            .expect("restore the loss scale");
        assert_eq!(t.optimizer_timestep(), start_step as u64);
        for _ in start_step..STEPS {
            let (loss, norm) = t.training_step(&model, &mut loader).expect("training step");
            losses.push(loss);
            norms.push(norm);
        }
    }
    Run {
        losses,
        norms,
        values: values_by_name_on(&parts, &client),
    }
}

/// Every loss finite, every gradient norm finite and non-zero, every
/// trained value finite.
fn assert_run_finite(label: &str, run: &Run) {
    for (step, (loss, norm)) in run.losses.iter().zip(&run.norms).enumerate() {
        assert!(loss.is_finite(), "{label} step {}: loss {loss}", step + 1);
        assert!(
            norm.is_finite() && *norm > 0.0,
            "{label} step {}: gradient norm {norm}",
            step + 1
        );
    }
    for (name, values) in &run.values {
        assert!(
            values.iter().all(|v| v.is_finite()),
            "{label}: {name} is not finite"
        );
    }
}

/// Per tensor, `max |got - want| <= rtol * max(max |want|, 1e-3)`.
fn assert_values_close(label: &str, got: &Values, want: &Values, rtol: f32) {
    let names = |v: &Values| v.keys().cloned().collect::<Vec<_>>();
    assert_eq!(names(got), names(want), "{label}: parameter names");
    for (name, w) in want {
        let g = &got[name];
        assert_eq!(g.len(), w.len(), "{label} {name}: length");
        let scale = w.iter().fold(0f32, |m, v| m.max(v.abs()));
        let tol = rtol * scale.max(1e-3);
        let mut worst = (0usize, 0f32);
        for (i, (a, b)) in g.iter().zip(w).enumerate() {
            assert!(a.is_finite(), "{label} {name}[{i}]: {a}");
            let d = (a - b).abs();
            if d > worst.1 {
                worst = (i, d);
            }
        }
        assert!(
            worst.1 <= tol,
            "{label} {name}: max abs diff {} at {} exceeds {tol} (got {}, want {})",
            worst.1,
            worst.0,
            g[worst.0],
            w[worst.0]
        );
    }
}

fn bits(values: &Values) -> BTreeMap<String, Vec<u32>> {
    values
        .iter()
        .map(|(name, v)| (name.clone(), v.iter().map(|x| x.to_bits()).collect()))
        .collect()
}

fn loss_bits(run: &Run) -> Vec<u64> {
    run.losses.iter().map(|l| l.to_bits()).collect()
}

#[test]
fn cuda_gdn_hybrid_matches_cpu_after_three_steps() {
    let Some((_guard, client, device)) = cuda() else {
        return;
    };
    let dir = scratch_dir("cuda-vs-cpu");
    let cfg = gdn_config(&dir);
    let init = write_init(&cfg, &dir);
    let want = fresh::<CpuRuntime>(&cfg, Some(&init), Stream::Shuffled, cpu(), STEPS, None);
    let got = fresh::<CudaRuntime>(
        &cfg,
        Some(&init),
        Stream::Shuffled,
        (client, device),
        STEPS,
        None,
    );
    assert_run_finite("cpu", &want);
    assert_run_finite("cuda", &got);
    assert!(
        got.values.contains_key("layers.0.linear_attn.A_log"),
        "the GDN mixer trains"
    );
    for (step, (g, w)) in got.losses.iter().zip(&want.losses).enumerate() {
        assert!(
            (g - w).abs() <= LOSS_RTOL * w.abs(),
            "step {}: CUDA loss {g} vs CPU loss {w}",
            step + 1
        );
    }
    assert_values_close("cuda vs cpu", &got.values, &want.values, PARAM_RTOL);
    let _ = fs::remove_dir_all(dir);
}

/// Every step scores the same batch, so each optimizer step lowers its loss.
#[test]
fn cuda_loss_falls_on_a_repeated_row() {
    let Some((_guard, client, device)) = cuda() else {
        return;
    };
    let dir = scratch_dir("cuda-loss");
    let cfg = gdn_config(&dir);
    let init = write_init(&cfg, &dir);
    let run = fresh::<CudaRuntime>(
        &cfg,
        Some(&init),
        Stream::Repeated,
        (client, device),
        STEPS,
        None,
    );
    assert_run_finite("cuda", &run);
    for (step, pair) in run.losses.windows(2).enumerate() {
        assert!(
            pair[1] < pair[0],
            "loss did not fall at step {}: {:?}",
            step + 2,
            run.losses
        );
    }
    let _ = fs::remove_dir_all(dir);
}

/// Bit equality needs a deterministic CUDA step. Two uninterrupted runs
/// check that first, and the resume check is skipped when they differ.
#[test]
fn cuda_resume_is_bit_identical_when_the_step_is_deterministic() {
    let Some((_guard, client, device)) = cuda() else {
        return;
    };
    let dir = scratch_dir("cuda-resume");
    let init_dir = scratch_dir("cuda-resume-init");
    let cfg = gdn_config(&dir);
    let init = write_init(&cfg, &init_dir);
    let backend = || (client.clone(), device.clone());
    let stream = Stream::Shuffled;
    let reference = fresh::<CudaRuntime>(&cfg, Some(&init), stream, backend(), STEPS, None);
    let again = fresh::<CudaRuntime>(&cfg, Some(&init), stream, backend(), STEPS, None);
    assert_run_finite("cuda", &reference);
    if bits(&reference.values) != bits(&again.values) || loss_bits(&reference) != loss_bits(&again)
    {
        eprintln!(
            "two uninterrupted CUDA runs differ, so the CUDA step is not deterministic: \
             skipping the bit-identical resume check"
        );
        let _ = fs::remove_dir_all(dir);
        let _ = fs::remove_dir_all(init_dir);
        return;
    }

    let save_at = Some(RESUME_AT);
    fresh::<CudaRuntime>(&cfg, Some(&init), stream, backend(), RESUME_AT, save_at);
    let (weights, meta, step) = find_latest_checkpoint(&cfg.trainer.checkpoint_dir)
        .expect("scan the checkpoint dir")
        .expect("the step set is eligible");
    assert_eq!(step, RESUME_AT);
    let resumed = resume::<CudaRuntime>(&cfg, stream, backend(), &weights, &meta, step);
    let reference_losses = loss_bits(&reference);
    assert_eq!(
        loss_bits(&resumed).as_slice(),
        &reference_losses[RESUME_AT..],
        "resumed step losses differ"
    );
    let (want, got) = (bits(&reference.values), bits(&resumed.values));
    assert_eq!(want.len(), got.len());
    for (name, b) in &want {
        assert_eq!(
            Some(b),
            got.get(name),
            "tensor '{name}' differs after resume"
        );
    }
    let _ = fs::remove_dir_all(dir);
    let _ = fs::remove_dir_all(init_dir);
}

/// Most steps the f16 case runs to reach `STEPS` clean steps.
#[cfg(feature = "f16")]
const F16_MAX_STEPS: usize = 24;

/// A `2^20` loss scale puts the f16 backward seed past f16's maximum, so the
/// first step overflows. Each overflow step skips the optimizer and halves
/// the scale. Once the scale fits, `STEPS` clean steps train.
#[cfg(feature = "f16")]
#[test]
fn cuda_f16_overflow_steps_skip_and_the_loss_stays_finite() {
    use boostr::optimizer::GradScalerState;
    use oxidizr::config::DTypePrecision;
    use oxidizr::loss_scale::LossScaleConfig;

    let Some((_guard, client, device)) = cuda() else {
        return;
    };
    let dir = scratch_dir("cuda-f16");
    let mut cfg = gdn_config(&dir);
    cfg.model.dtype = DTypePrecision::F16;
    cfg.trainer.loss_scale = Some(LossScaleConfig {
        initial: 1048576.0,
        ..LossScaleConfig::default()
    });
    let backend = (client.clone(), device.clone());
    let model = model_on::<CudaRuntime>(&cfg, None, &backend);
    let mut parts = train_parts_on(&model);
    let mut loader = loader_on::<CudaRuntime>(&cfg, Stream::Repeated, device);
    let layout = loader.row_layout().expect("row layout");
    loader.seek(DataPosition::default()).expect("seek");
    let weight_decay = resolve_weight_decay(&cfg.trainer, None, true).expect("weight decay rule");
    let mut clean = 0;
    {
        let start = (0, DataPosition::default());
        let mut t = trainer_on(&cfg, &mut parts, layout, start, weight_decay, backend);
        let (loss, norm) = t
            .training_step(&model, &mut loader)
            .expect("first f16 step");
        assert!(
            !norm.is_finite(),
            "the first step must overflow: norm {norm}"
        );
        assert!(loss.is_finite(), "the reported loss is unscaled: {loss}");
        assert_eq!(t.optimizer_timestep(), 0, "an overflow step must not count");
        assert_eq!(
            t.loss_scale_state(),
            Some(GradScalerState {
                scale: 524288.0,
                consecutive_ok: 0
            })
        );
        for step in 2..=F16_MAX_STEPS {
            let before = t.loss_scale_state().expect("an f16 run scales its loss");
            let timestep = t.optimizer_timestep();
            let (loss, norm) = t.training_step(&model, &mut loader).expect("f16 step");
            let after = t.loss_scale_state().expect("an f16 run scales its loss");
            assert!(loss.is_finite(), "step {step}: loss {loss}");
            if norm.is_finite() {
                clean += 1;
                assert_eq!(t.optimizer_timestep(), timestep + 1, "step {step}: clean");
                assert_eq!(after.scale, before.scale, "step {step}: the scale holds");
            } else {
                assert_eq!(t.optimizer_timestep(), timestep, "step {step}: overflow");
                assert_eq!(
                    after.scale,
                    before.scale * 0.5,
                    "step {step}: the scale halves"
                );
            }
            if clean == STEPS {
                break;
            }
        }
    }
    assert_eq!(
        clean, STEPS,
        "no {STEPS} clean f16 steps within {F16_MAX_STEPS}"
    );
    for (name, values) in values_by_name_on(&parts, &client) {
        assert!(values.iter().all(|v| v.is_finite()), "{name} is not finite");
    }
    let _ = fs::remove_dir_all(dir);
}

/// bf16 needs no loss scale. Three steps keep every loss and norm finite.
#[cfg(feature = "f16")]
#[test]
fn cuda_bf16_steps_stay_finite() {
    use oxidizr::config::DTypePrecision;

    let Some((_guard, client, device)) = cuda() else {
        return;
    };
    let dir = scratch_dir("cuda-bf16");
    let mut cfg = gdn_config(&dir);
    cfg.model.dtype = DTypePrecision::BF16;
    let run = fresh::<CudaRuntime>(&cfg, None, Stream::Repeated, (client, device), STEPS, None);
    assert_run_finite("bf16", &run);
    let _ = fs::remove_dir_all(dir);
}
