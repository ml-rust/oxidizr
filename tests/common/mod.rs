//! Shared helpers for the integration tests: a tiny CPU model, its trainer, and scratch dirs.

// Each test binary compiles this module and uses a subset of it.
#![allow(dead_code)]

use boostr::tensor::TensorId;
use boostr::{CpuClient, CpuDevice, CpuRuntime, DType, Tensor, TypeConversionOps, VarMap};
use oxidizr::config::{
    DTypePrecision, DistributedConfig, LitConfig, OxidizrModelConfig, TargetDevice, TrainerConfig,
};
use oxidizr::data::{DataPosition, LitDataLoader, RowLayout, RowOrder};
use oxidizr::model::LitGPT;
use oxidizr::schedule::LrSchedule;
use oxidizr::trainer::{LitTrainer, TrainerRunConfig};
use oxidizr::weight_decay::{resolve_weight_decay, WeightDecaySpec};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

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
    let mut varmap = VarMap::<CpuRuntime>::new();
    LitGPT::<CpuRuntime>::new_seeded(cfg, &mut varmap, &device, &client).expect("build model")
}

/// The optimizer inputs `commands::train` derives from a model.
pub struct TrainParts {
    pub ids: Vec<TensorId>,
    pub params: HashMap<TensorId, Tensor<CpuRuntime>>,
    pub names: Vec<(String, TensorId)>,
}

pub fn train_parts(model: &LitGPT<CpuRuntime>) -> TrainParts {
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
            let values = tensor.to_vec::<f32>();
            (name.clone(), values.iter().map(|v| v.to_bits()).collect())
        })
        .collect()
}
