//! Shared helpers for the integration tests: a tiny CPU model, its trainer, and scratch dirs.

use boostr::tensor::TensorId;
use boostr::{CpuClient, CpuDevice, CpuRuntime, Tensor, VarMap};
use oxidizr::config::{
    DTypePrecision, DistributedConfig, LitConfig, OxidizrModelConfig, TargetDevice, TrainerConfig,
};
use oxidizr::data::{DataPosition, LitDataLoader, RowLayout, RowOrder};
use oxidizr::model::LitGPT;
use oxidizr::schedule::LrSchedule;
use oxidizr::trainer::{LitTrainer, TrainerRunConfig};
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
            z_loss_alpha: 0.0,
            warmup_steps: Some(2),
            warmup_ratio: None,
            lr_schedule: LrSchedule::Cosine,
            min_lr_ratio: 0.1,
            target_device: TargetDevice::Cpu,
            distributed: DistributedConfig::default(),
            seed: Some(42),
            target_mask: None,
            shuffle_rows: true,
            group_size: 4,
            beta_kl: 0.04,
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
    let vocab = cfg.model.vocab_size as u32;
    let tokens: Vec<u32> = (0..4000u32).map(|i| (i * 7 + i / 5) % vocab).collect();
    LitDataLoader::<CpuRuntime>::new(
        tokens,
        cfg.trainer.batch_size,
        cfg.model.max_seq_len,
        CpuDevice::new(),
    )
    .with_row_order(RowOrder::Shuffled { seed: 7 })
}

/// A trainer over `params` that starts at `start_step` and `start_position`.
pub fn trainer<'a>(
    cfg: &LitConfig,
    parts: &'a mut TrainParts,
    row_layout: RowLayout,
    start_step: usize,
    start_position: DataPosition,
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
        },
    )
    .expect("build trainer")
}

/// Bit patterns of every trainable tensor, keyed by parameter name.
pub fn bits_by_name(parts: &TrainParts) -> BTreeMap<String, Vec<u32>> {
    parts
        .names
        .iter()
        .map(|(name, id)| {
            let values = parts.params[id]
                .contiguous()
                .expect("contiguous")
                .to_vec::<f32>();
            (name.clone(), values.iter().map(|v| v.to_bits()).collect())
        })
        .collect()
}
