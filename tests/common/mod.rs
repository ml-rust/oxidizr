//! Shared helpers for the integration tests: a tiny model, its trainer, and scratch dirs.
//!
//! The plain helpers build on the CPU runtime. The `_on` helpers take any
//! runtime whose client satisfies [`TrainClient`].

// Each test binary compiles this module and uses a subset of it.
#![allow(dead_code)]

use boostr::ops::FusedOptimizerOps;
use boostr::tensor::TensorId;
use boostr::{
    BinaryOps, CompareOps, ConvOps, CpuClient, CpuDevice, CpuRuntime, DType, DequantOps,
    FlashAttentionOps, GatedDeltaNetOps, IndexingOps, NormalizationOps, QuantMatmulOps, RandomOps,
    ReduceOps, RoPEOps, Runtime, RuntimeClient, ScalarOps, ShapeOps, Tensor, TensorOps,
    TypeConversionOps, UnaryOps, VarMap,
};
use oxidizr::checkpoint_set::plan_optimizer_resume;
use oxidizr::commands::train_data::{run_loaders, TokenSource};
use oxidizr::config::{
    find_latest_checkpoint, ActivationCheckpointing, CheckpointMeta, DTypePrecision,
    DistributedConfig, GdnSpec, LitConfig, LoraConfig, OxidizrModelConfig, TargetDevice,
    TrainerConfig,
};
use oxidizr::data::{DataPosition, LitDataLoader, RowLayout, RowOrder};
use oxidizr::eval::{read_eval_resume, EvalConfig, Evaluator};
use oxidizr::fit::{CheckpointNote, EvalReport, FitStart, FitSummary, StepReport, StepSink};
use oxidizr::loss_scale::read_loss_scale_state;
use oxidizr::model::{load_checkpoint_into_model, LitGPT};
use oxidizr::prelude::{Error, Result};
use oxidizr::schedule::LrSchedule;
use oxidizr::trainer::{LitTrainer, TrainerRunConfig};
use oxidizr::weight_decay::{inherited_weight_decay, resolve_weight_decay, WeightDecaySpec};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// Client ops a training run needs: the bounds `commands::train` puts on its
/// runtime, plus the ones a checkpoint load adds.
pub trait TrainClient<R: Runtime<DType = DType>>:
    RuntimeClient<R>
    + TensorOps<R>
    + FusedOptimizerOps<R>
    + RandomOps<R>
    + ScalarOps<R>
    + BinaryOps<R>
    + CompareOps<R>
    + ConvOps<R>
    + FlashAttentionOps<R>
    + IndexingOps<R>
    + NormalizationOps<R>
    + QuantMatmulOps<R>
    + RoPEOps<R>
    + TypeConversionOps<R>
    + DequantOps<R>
    + GatedDeltaNetOps<R>
    + UnaryOps<R>
    + ReduceOps<R>
    + ShapeOps<R>
{
}

impl<R, C> TrainClient<R> for C
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + TensorOps<R>
        + FusedOptimizerOps<R>
        + RandomOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConvOps<R>
        + FlashAttentionOps<R>
        + IndexingOps<R>
        + NormalizationOps<R>
        + QuantMatmulOps<R>
        + RoPEOps<R>
        + TypeConversionOps<R>
        + DequantOps<R>
        + GatedDeltaNetOps<R>
        + UnaryOps<R>
        + ReduceOps<R>
        + ShapeOps<R>,
{
}

/// A two-layer dense model small enough to train in a unit-sized test.
pub fn tiny_model_config(tie_word_embeddings: bool) -> OxidizrModelConfig {
    OxidizrModelConfig {
        name: "resume-test".to_string(),
        hidden_size: 16,
        num_layers: 2,
        num_heads: 4,
        head_dim: None,
        kv_heads: None,
        kv_latent_dim: None,
        q_latent_dim: None,
        d_rope: None,
        num_experts: None,
        experts_per_tok: None,
        shared_expert_enabled: None,
        intermediate_size: Some(32),
        mamba_layers: None,
        mamba2_num_heads: None,
        mamba2_head_dim: None,
        mamba2_state_size: None,
        mamba2_chunk_size: None,
        mamba2_n_groups: None,
        mamba2_conv_kernel: None,
        mamba2_expand: None,
        mamba3_enabled: None,
        mamba3_complex_rope: None,
        mamba3_mimo_rank: None,
        mamba3_use_conv: None,
        gdn_layers: None,
        gdn: None,
        vocab_size: 24,
        grow_vocab: false,
        tie_word_embeddings,
        tokenizer_vocab: "custom".to_string(),
        max_seq_len: 8,
        rope_theta: 10000.0,
        rope_scaling: None,
        sliding_window: None,
        qk_norm: false,
        speech_layout: None,
        lora: None,
        dtype: DTypePrecision::F32,
    }
}

/// [`tiny_model_config`] with Gated DeltaNet layer 0 and attention layer 1.
///
/// Chunks of 4 over 8 tokens run the chunked recurrence with two chunks.
pub fn tiny_gdn_model_config() -> OxidizrModelConfig {
    OxidizrModelConfig {
        name: "gdn-test".to_string(),
        kv_heads: Some(2),
        gdn_layers: Some(vec![0]),
        gdn: Some(GdnSpec {
            state_size: 4,
            key_heads: 2,
            value_heads: 4,
            conv_kernel: 4,
            chunk_size: 4,
            rms_eps: 1e-6,
        }),
        ..tiny_model_config(false)
    }
}

/// A seeded config with warmup, cosine decay, clipping and two micro-batches per step.
pub fn tiny_config(checkpoint_dir: &Path, max_steps: usize) -> LitConfig {
    LitConfig {
        model: tiny_model_config(false),
        trainer: TrainerConfig {
            learning_rate: 3e-3,
            batch_size: 2,
            max_steps,
            num_epochs: 1,
            gradient_accumulation: 2,
            checkpoint_dir: checkpoint_dir.to_string_lossy().into_owned(),
            checkpoint_sidecars: Vec::new(),
            log_interval: 10,
            save_interval: 0,
            max_checkpoints: 0,
            async_checkpoint: false,
            load_balance_alpha: 0.0,
            max_grad_norm: 1.0,
            target_grad_norm: None,
            max_grad_value: None,
            adamw_beta1: 0.9,
            adamw_beta2: 0.95,
            adamw_eps: 1e-6,
            weight_decay: 0.1,
            weight_decay_rule: None,
            no_decay_patterns: None,
            z_loss_alpha: 0.0,
            warmup_steps: Some(2),
            warmup_ratio: None,
            lr_schedule: LrSchedule::Cosine,
            min_lr_ratio: 0.1,
            decay_ratio: None,
            decay_steps: None,
            decay_shape: None,
            target_device: TargetDevice::Cpu,
            distributed: DistributedConfig::default(),
            seed: Some(42),
            target_mask: None,
            shuffle_rows: true,
            loss_scale: None,
            eval: None,
            metrics: None,
            spike_guard: None,
            activation_checkpointing: ActivationCheckpointing::None,
        },
    }
}

/// An empty directory under Cargo's per-target scratch directory.
pub fn scratch_dir(tag: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("oxidizr-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// The deterministic token stream of [`data_loader`], `count` tokens long.
pub fn tokens(cfg: &LitConfig, count: u32) -> Vec<u32> {
    let vocab = cfg.model.vocab_size as u32;
    (0..count).map(|i| (i * 7 + i / 5) % vocab).collect()
}

/// Writes `tokens` as a raw little-endian `u32` data file at `path`.
pub fn write_token_file(path: &Path, tokens: &[u32]) {
    let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    std::fs::write(path, bytes).expect("write token file");
}

/// Writes `cfg` as a YAML config file at `path`, as `oxidizr train -f` reads it.
pub fn write_config_yaml(path: &Path, cfg: &LitConfig) {
    let yaml = serde_yaml::to_string(cfg).expect("serialize config");
    std::fs::write(path, yaml).expect("write config yaml");
}

/// Build the model as `commands::train` does: a fresh `VarMap` and `LitGPT::new_seeded`.
pub fn build_model(cfg: &LitConfig) -> LitGPT<CpuRuntime> {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    build_model_on::<CpuRuntime>(cfg, &device, &client)
}

/// [`build_model`] on `device`, drawing on `client`.
pub fn build_model_on<R>(cfg: &LitConfig, device: &R::Device, client: &R::Client) -> LitGPT<R>
where
    R: Runtime<DType = DType>,
    R::Client: TrainClient<R>,
{
    let mut varmap = VarMap::<R>::new();
    LitGPT::<R>::new_seeded(cfg, &mut varmap, device, client).expect("build model")
}

/// The optimizer inputs `commands::train` derives from a model.
pub struct TrainParts<R: Runtime<DType = DType> = CpuRuntime> {
    pub ids: Vec<TensorId>,
    pub params: HashMap<TensorId, Tensor<R>>,
    pub names: Vec<(String, TensorId)>,
}

pub fn train_parts(model: &LitGPT<CpuRuntime>) -> TrainParts {
    train_parts_on(model)
}

/// [`train_parts`] for a model on any runtime.
pub fn train_parts_on<R: Runtime<DType = DType>>(model: &LitGPT<R>) -> TrainParts<R> {
    let trainable = model.trainable_parameters();
    let ids = trainable.iter().map(|(id, _)| *id).collect();
    let params = trainable
        .into_iter()
        .map(|(id, var)| (id, var.tensor().clone()))
        .collect();
    let names = model
        .named_trainable_parameters()
        .expect("named parameters")
        .into_iter()
        .map(|(name, id, _)| (name, id))
        .collect();
    TrainParts { ids, params, names }
}

/// A deterministic token stream, read in a seeded shuffled row order.
pub fn data_loader(cfg: &LitConfig) -> LitDataLoader<CpuRuntime> {
    LitDataLoader::<CpuRuntime>::new(
        tokens(cfg, 4000),
        cfg.trainer.batch_size,
        cfg.model.max_seq_len,
        CpuDevice::new(),
    )
    .with_row_order(RowOrder::Shuffled { seed: 7 })
}

/// A held-out eval of `rows` rows every `interval` steps, `batches` batches each.
pub fn holdout_eval(rows: usize, interval: usize, batches: usize) -> EvalConfig {
    EvalConfig {
        data_file: None,
        holdout_rows: Some(rows),
        holdout_fraction: None,
        interval,
        batches,
        early_stopping: None,
    }
}

/// The training loader and eval of `cfg`, split by the code `commands::train` runs.
///
/// The tokens are those of [`data_loader`], and training reads them in its
/// shuffled row order. `cfg.trainer.eval` must be set.
pub fn eval_loaders(cfg: &LitConfig) -> (LitDataLoader<CpuRuntime>, Evaluator<CpuRuntime>) {
    let source = TokenSource::Generated(tokens(cfg, 4000));
    let loaders = run_loaders::<CpuRuntime>(cfg, &source, &CpuDevice::new()).expect("loaders");
    let train = loaders.train.with_row_order(RowOrder::Shuffled { seed: 7 });
    (train, loaders.evaluator.expect("trainer.eval is set"))
}

/// A trainer over `params` that starts at `start_step` and `start_position`.
///
/// It uses the fresh-run weight decay rule of `cfg`.
pub fn trainer<'a>(
    cfg: &LitConfig,
    parts: &'a mut TrainParts,
    row_layout: RowLayout,
    start_step: usize,
    start_position: DataPosition,
) -> LitTrainer<'a, CpuRuntime> {
    let weight_decay = resolve_weight_decay(&cfg.trainer, None, true).expect("weight decay rule");
    trainer_with_decay(
        cfg,
        parts,
        row_layout,
        start_step,
        start_position,
        weight_decay,
    )
}

/// [`trainer`] with the resolved weight decay rule `weight_decay`.
pub fn trainer_with_decay<'a>(
    cfg: &LitConfig,
    parts: &'a mut TrainParts,
    row_layout: RowLayout,
    start_step: usize,
    start_position: DataPosition,
    weight_decay: WeightDecaySpec,
) -> LitTrainer<'a, CpuRuntime> {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    trainer_on(
        cfg,
        parts,
        row_layout,
        (start_step, start_position),
        weight_decay,
        (client, device),
    )
}

/// [`trainer_with_decay`] on any runtime: it trains on `backend`'s
/// `(client, device)` and starts at `start`'s `(step, position)`.
pub fn trainer_on<'a, R>(
    cfg: &LitConfig,
    parts: &'a mut TrainParts<R>,
    row_layout: RowLayout,
    start: (usize, DataPosition),
    weight_decay: WeightDecaySpec,
    backend: (R::Client, R::Device),
) -> LitTrainer<'a, R>
where
    R: Runtime<DType = DType>,
    R::Client: TrainClient<R>,
{
    let (start_step, start_position) = start;
    let (client, device) = backend;
    let steps_json_path = format!("{}/steps.json", cfg.trainer.checkpoint_dir);
    LitTrainer::new(
        cfg.trainer.clone(),
        &mut parts.params,
        parts.ids.clone(),
        parts.names.clone(),
        client,
        device,
        TrainerRunConfig {
            row_layout,
            start_position,
            batch_size: cfg.trainer.batch_size,
            steps_json_path,
            headless: true,
            start_step,
            model_config: cfg.model.clone(),
            data_file_path: None,
            weight_decay,
        },
    )
    .expect("build trainer")
}

/// Bit patterns of every trainable tensor, keyed by parameter name.
///
/// A half-precision tensor is widened to f32 first. The cast is exact, so
/// equal bits still mean equal tensors.
pub fn bits_by_name(parts: &TrainParts) -> BTreeMap<String, Vec<u32>> {
    let client = CpuClient::new(CpuDevice::new());
    bits_by_name_on(parts, &client)
}

/// [`bits_by_name`] for parameters on `client`'s runtime.
pub fn bits_by_name_on<R>(parts: &TrainParts<R>, client: &R::Client) -> BTreeMap<String, Vec<u32>>
where
    R: Runtime<DType = DType>,
    R::Client: TypeConversionOps<R>,
{
    values_by_name_on(parts, client)
        .into_iter()
        .map(|(name, values)| (name, values.iter().map(|v| v.to_bits()).collect()))
        .collect()
}

/// Host values of every trainable tensor, keyed by parameter name.
///
/// A half-precision tensor is widened to f32 on `client` first. The cast is exact.
pub fn values_by_name_on<R>(parts: &TrainParts<R>, client: &R::Client) -> BTreeMap<String, Vec<f32>>
where
    R: Runtime<DType = DType>,
    R::Client: TypeConversionOps<R>,
{
    parts
        .names
        .iter()
        .map(|(name, id)| {
            let tensor = parts.params[id].contiguous().expect("contiguous");
            let tensor = if tensor.dtype() == DType::F32 {
                tensor
            } else {
                client.cast(&tensor, DType::F32).expect("widen to f32")
            };
            (name.clone(), tensor.to_vec::<f32>())
        })
        .collect()
}

/// Keys of a `steps.json` line that depend on wall-clock time.
pub const TIMING_KEYS: [&str; 2] = ["tokens_per_sec", "mfu"];

/// The lines of `steps_json` without [`TIMING_KEYS`], so two runs compare equal.
pub fn steps_without_timing(steps_json: &str) -> Vec<serde_json::Value> {
    steps_json
        .lines()
        .map(|line| {
            let mut value: serde_json::Value =
                serde_json::from_str(line).expect("a steps.json line");
            if let Some(map) = value.as_object_mut() {
                for key in TIMING_KEYS {
                    map.remove(key);
                }
            }
            value
        })
        .collect()
}

/// TensorBoard event files in `dir`.
pub fn event_files(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .map(|it| {
            it.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("events.out.tfevents."))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// LoRA adapters of rank 2 on `q_proj` and `v_proj`, with the whole base frozen.
pub fn tiny_lora() -> LoraConfig {
    LoraConfig {
        rank: 2,
        alpha: 4.0,
        target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        freeze_base: true,
        train_modules: Vec::new(),
    }
}

/// What one [`fit_session`] reached.
pub struct FitSession {
    /// Bit patterns of every trainable tensor after the session.
    pub bits: BTreeMap<String, Vec<u32>>,
    pub final_step: usize,
    pub stop_reason: Option<String>,
}

/// Fits `cfg` from the latest checkpoint in its directory, or from step 0 when
/// `resume` is false, as `commands::train` sets the run up.
///
/// The latest checkpoint is the one `--resume auto` picks. An eval runs when
/// `cfg.trainer.eval` is set.
pub fn fit_session(
    cfg: &LitConfig,
    resume: bool,
    sink: &mut dyn StepSink,
) -> (FitSession, Result<()>) {
    let mut model = build_model(cfg);
    let (start_step, found) = if resume {
        let (weights, meta, step) = find_latest_checkpoint(&cfg.trainer.checkpoint_dir)
            .expect("scan")
            .expect("a committed step set");
        load_checkpoint_into_model(&mut model, &weights).expect("load weights");
        (step, Some((weights, meta)))
    } else {
        (0, None)
    };
    let weights = found.as_ref().map(|(w, _)| w.as_str());
    let meta = found.as_ref().map(|(_, m)| m.as_str());
    let plan = plan_optimizer_resume(weights, meta, start_step).expect("optimizer plan");
    let inherited = inherited_weight_decay(meta, start_step).expect("inherited rule");
    let decay = resolve_weight_decay(&cfg.trainer, inherited.as_ref(), true).expect("rule");
    let scale = read_loss_scale_state(meta, start_step).expect("loss scale");

    let (mut train, evaluator) = match cfg.trainer.eval {
        Some(_) => {
            let (train, evaluator) = eval_loaders(cfg);
            let state =
                read_eval_resume(meta, start_step, evaluator.holdout_rows()).expect("eval state");
            (train, Some(evaluator.resumed(state)))
        }
        None => (data_loader(cfg), None),
    };
    let layout = train.row_layout().expect("row layout");
    let position = meta
        .map(|m| CheckpointMeta::load(m).expect("meta"))
        .and_then(|m| m.data_position)
        .unwrap_or_default();
    train.seek(position).expect("seek");

    let mut parts = train_parts(&model);
    let (final_step, stop_reason, result) = {
        let mut t = trainer_with_decay(cfg, &mut parts, layout, start_step, position, decay);
        t.apply_optimizer_resume(&plan).expect("optimizer state");
        t.restore_loss_scale(scale).expect("loss scale");
        if let Some(evaluator) = evaluator {
            t.attach_eval(evaluator);
        }
        let result = t.fit_with_sink(&model, &mut train, sink);
        (
            t.global_step(),
            t.early_stop_reason().map(str::to_string),
            result,
        )
    };
    let session = FitSession {
        bits: bits_by_name(&parts),
        final_step,
        stop_reason,
    };
    (session, result)
}

/// A sink that ends the run with an error in `on_step` of step `at`.
///
/// The step's eval and save never run. The fit loop then waits for every
/// checkpoint write in flight, so the sets before step `at` are on disk.
pub struct StopAtStep<S> {
    pub inner: S,
    pub at: usize,
}

impl<S: StepSink> StepSink for StopAtStep<S> {
    fn on_start(&mut self, start: &FitStart) -> Result<()> {
        self.inner.on_start(start)
    }
    fn on_step(&mut self, report: &StepReport) -> Result<()> {
        self.inner.on_step(report)?;
        if report.step == self.at {
            return Err(Error::TrainingError {
                reason: format!("stopped at step {}", self.at),
            });
        }
        Ok(())
    }
    fn on_eval(&mut self, step: usize, report: &EvalReport) -> Result<()> {
        self.inner.on_eval(step, report)
    }
    fn on_checkpoint(&mut self, note: &CheckpointNote) -> Result<()> {
        self.inner.on_checkpoint(note)
    }
    fn on_error(&mut self, step: usize, error: &Error) {
        self.inner.on_error(step, error)
    }
    fn on_finish(&mut self, summary: &FitSummary) -> Result<()> {
        self.inner.on_finish(summary)
    }
}

/// Sorted entry names of `dir`, without TensorBoard event files.
///
/// Only a `tensorboard` build writes event files, and their names hold a time.
pub fn dir_entries(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("read dir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| !name.starts_with("events.out.tfevents."))
        .collect();
    names.sort();
    names
}

/// The checkpoint meta at `path` as JSON, without what differs between two runs.
///
/// `timestamp` goes, and `checkpoint_file` keeps its file name only.
pub fn comparable_meta(path: &Path) -> serde_json::Value {
    let text = std::fs::read_to_string(path).expect("read meta");
    let mut meta: serde_json::Value = serde_json::from_str(&text).expect("meta json");
    let map = meta.as_object_mut().expect("meta object");
    map.remove("timestamp");
    let file = map
        .get("checkpoint_file")
        .and_then(serde_json::Value::as_str)
        .expect("checkpoint_file")
        .to_string();
    let name = Path::new(&file)
        .file_name()
        .expect("file name")
        .to_string_lossy()
        .into_owned();
    map.insert(
        "checkpoint_file".to_string(),
        serde_json::Value::String(name),
    );
    meta
}

/// `a` and `b` hold the same checkpoint files, byte-equal, with equal metas.
///
/// Every `.safetensors` file must match byte for byte. Every `.meta.json`
/// must match after [`comparable_meta`]. Returns the checkpoint file names.
pub fn assert_same_checkpoints(a: &Path, b: &Path) -> Vec<String> {
    let names = dir_entries(a);
    assert_eq!(names, dir_entries(b), "the checkpoint files differ");
    let mut checked = Vec::new();
    for name in names {
        if name.ends_with(".safetensors") {
            let left = std::fs::read(a.join(&name)).expect("read file");
            let right = std::fs::read(b.join(&name)).expect("read file");
            assert!(left == right, "{name} differs between the runs");
            checked.push(name);
        } else if name.ends_with(".meta.json") {
            assert_eq!(
                comparable_meta(&a.join(&name)),
                comparable_meta(&b.join(&name)),
                "{name} differs between the runs"
            );
            checked.push(name);
        }
    }
    checked
}

/// GPT-2's byte-to-character map. Entry `b` spells byte `b` in a byte-level vocabulary.
pub fn byte_level_alphabet() -> Vec<char> {
    let printable = |b: u32| {
        (u32::from(b'!')..=u32::from(b'~')).contains(&b)
            || (0xA1..=0xAC).contains(&b)
            || (0xAE..=0xFF).contains(&b)
    };
    let mut shifted = 0u32;
    (0u32..256)
        .map(|b| {
            let code = if printable(b) {
                b
            } else {
                shifted += 1;
                255 + shifted
            };
            char::from_u32(code).expect("a valid code point")
        })
        .collect()
}

/// Id of `<|endoftext|>` in [`write_byte_level_tokenizer`]: its EOS.
pub const BYTE_LEVEL_EOS: u32 = 256;
/// Id of `<s>` in [`write_byte_level_tokenizer`] with `bos`.
pub const BYTE_LEVEL_BOS: u32 = 257;

/// Writes a byte-level BPE `tokenizer.json` into `dir` and returns its path.
///
/// Byte `b` is id `b`. One merge makes `th` id 258. `<|endoftext|>` is id
/// [`BYTE_LEVEL_EOS`]. With `bos`, a template places `<s>`
/// ([`BYTE_LEVEL_BOS`]) before every sequence.
pub fn write_byte_level_tokenizer(dir: &Path, bos: bool) -> PathBuf {
    let alphabet = byte_level_alphabet();
    let mut vocab = serde_json::Map::new();
    for (id, ch) in alphabet.iter().enumerate() {
        vocab.insert(ch.to_string(), serde_json::json!(id));
    }
    vocab.insert("th".to_string(), serde_json::json!(258));
    let mut added = vec![serde_json::json!(
        {"id": BYTE_LEVEL_EOS, "content": "<|endoftext|>", "special": true}
    )];
    let mut json = serde_json::json!({
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
        "model": {"type": "BPE", "vocab": vocab, "merges": [["t", "h"]]}
    });
    if bos {
        added.push(serde_json::json!({"id": BYTE_LEVEL_BOS, "content": "<s>", "special": true}));
        json["post_processor"] = serde_json::json!({
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}}
            ],
            "pair": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "<s>", "type_id": 1}},
                {"Sequence": {"id": "B", "type_id": 1}}
            ],
            "special_tokens": {"<s>": {"id": "<s>", "ids": [BYTE_LEVEL_BOS], "tokens": ["<s>"]}}
        });
    }
    json["added_tokens"] = serde_json::Value::Array(added);
    let name = if bos {
        "bos-tokenizer.json"
    } else {
        "tokenizer.json"
    };
    let path = dir.join(name);
    std::fs::write(&path, json.to_string()).expect("write tokenizer.json");
    path
}

/// The little-endian `u32` ids of the token file at `path`.
pub fn read_tokens_le(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).expect("read token file");
    assert_eq!(bytes.len() % 4, 0, "a token file holds whole u32 ids");
    let (ids, _) = bytes.as_chunks::<4>();
    ids.iter().map(|c| u32::from_le_bytes(*c)).collect()
}
