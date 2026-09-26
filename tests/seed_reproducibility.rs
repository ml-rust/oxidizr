//! `TrainerConfig::seed` must produce reproducible initial weights through
//! the exact model-construction path `commands::train::run_train_with_runtime`
//! uses: a fresh `VarMap` handed to `LitGPT::new_seeded`, which applies
//! `VarBuilder::with_seed` before calling `LitGPT::new`. Every test here
//! builds through `LitGPT::new_seeded` (never a hand-rolled `VarBuilder`), so
//! a regression in the `commands/train.rs` wiring or in `new_seeded` itself
//! shows up here, not just in a unit test of `VarBuilder` alone.

use boostr::{CpuClient, CpuDevice, CpuRuntime, VarMap};
use oxidizr::config::{
    DTypePrecision, DistributedConfig, LitConfig, OxidizrModelConfig, TargetDevice, TrainerConfig,
};
use oxidizr::model::LitGPT;

fn tiny_model_config() -> OxidizrModelConfig {
    OxidizrModelConfig {
        name: "seed-test".to_string(),
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
        tie_word_embeddings: false,
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

fn trainer_config(seed: Option<u64>) -> TrainerConfig {
    TrainerConfig {
        learning_rate: 3e-4,
        batch_size: 1,
        max_steps: 1,
        num_epochs: 1,
        gradient_accumulation: 1,
        checkpoint_dir: "./checkpoints".to_string(),
        log_interval: 10,
        save_interval: 500,
        max_checkpoints: 10,
        load_balance_alpha: 0.0,
        max_grad_norm: 1.0,
        target_grad_norm: None,
        max_grad_value: None,
        adamw_beta1: 0.9,
        adamw_beta2: 0.95,
        adamw_eps: 1e-6,
        weight_decay: 0.1,
        z_loss_alpha: 0.0,
        warmup_steps: None,
        warmup_ratio: Some(0.1),
        lr_schedule: oxidizr::schedule::LrSchedule::Constant,
        min_lr_ratio: 0.0,
        target_device: TargetDevice::Cpu,
        distributed: DistributedConfig::default(),
        seed,
        target_mask: None,
        group_size: 4,
        beta_kl: 0.04,
    }
}

fn config(seed: Option<u64>) -> LitConfig {
    LitConfig {
        model: tiny_model_config(),
        trainer: trainer_config(seed),
    }
}

/// Build a model the same way `run_train_with_runtime` does: a fresh
/// `VarMap` passed to `LitGPT::new_seeded`. This is the call site that
/// carries `config.trainer.seed` into `VarBuilder::with_seed` — if that
/// wiring in `LitGPT::new_seeded` (or its use from `commands/train.rs`) were
/// reverted, every seeded assertion below would fail.
fn build(cfg: &LitConfig) -> LitGPT<CpuRuntime> {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let mut varmap = VarMap::<CpuRuntime>::new();
    LitGPT::<CpuRuntime>::new_seeded(cfg, &mut varmap, &device, &client)
        .expect("seeded model construction must succeed")
}

/// Every trainable tensor's name and values, in `named_parameters` order.
fn named_values(model: &LitGPT<CpuRuntime>) -> Vec<(String, Vec<f32>)> {
    model
        .named_parameters()
        .expect("named_parameters must succeed on a dense model")
        .into_iter()
        .map(|(name, _id, var)| {
            let values = var
                .tensor()
                .contiguous()
                .expect("contiguous")
                .to_vec::<f32>();
            (name, values)
        })
        .collect()
}

/// The discriminating assertion: two independently constructed models, same
/// seed, must have byte-identical initial weights across several parameter
/// tensors (not just one, not just shapes). This exercises `LitGPT::new_seeded`
/// end to end, so reverting the `with_seed` call added to `commands/train.rs`
/// (or removing it from `new_seeded`) makes `VarBuilder::seed` stay `None`,
/// `take_or_init_tensor` falls back to unseeded `Init::init_tensor`, and the
/// two builds diverge — this test then fails.
#[test]
fn same_seed_produces_identical_initial_weights() {
    let cfg = config(Some(42));
    let a = named_values(&build(&cfg));
    let b = named_values(&build(&cfg));

    assert_eq!(a.len(), b.len(), "same config must expose the same tensors");
    assert!(
        a.len() >= 4,
        "need several parameter tensors to compare, got {}",
        a.len()
    );
    for ((name_a, va), (name_b, vb)) in a.iter().zip(b.iter()) {
        assert_eq!(name_a, name_b, "tensor order must match between builds");
        assert_eq!(
            va, vb,
            "tensor '{name_a}' differs between two same-seed builds"
        );
    }
}

#[test]
fn different_seeds_produce_different_initial_weights() {
    let a = named_values(&build(&config(Some(1))));
    let b = named_values(&build(&config(Some(2))));

    assert_eq!(a.len(), b.len());
    let differing = a
        .iter()
        .zip(b.iter())
        .filter(|((_, va), (_, vb))| va != vb)
        .count();
    assert!(
        differing >= a.len() / 2,
        "most parameter tensors should differ between distinct seeds, \
         only {differing} of {} did",
        a.len()
    );
}

#[test]
fn unset_seed_still_builds_finite_weights() {
    let model = build(&config(None));
    for (name, values) in named_values(&model) {
        assert!(
            values.iter().all(|v| v.is_finite()),
            "tensor '{name}' has a non-finite value when seed is None"
        );
    }
}
