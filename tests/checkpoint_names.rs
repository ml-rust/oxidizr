//! Golden-name safety net for `LitGPT::extend_block_named_trainable`
//! (`src/model.rs`, dense-attention arm around line 4391).
//!
//! `load_checkpoint_into_model` keys every tensor by NAME and hard-errors on
//! a missing one (see `src/model.rs` around `missing.is_empty()`). A single
//! renamed parameter in the dense-attention block therefore makes every
//! existing checkpoint unloadable. This file hardcodes the exact name set for
//! each dense configuration so an upcoming refactor of that block cannot
//! rename a parameter without a test failing here first.
//!
//! Golden lists are written out literally, not generated from
//! `named_parameters` itself — generating them from the function under test
//! would make the assertion vacuous.

use std::collections::{BTreeSet, HashMap};

use boostr::tensor::TensorId;
use boostr::{
    CpuClient, CpuDevice, CpuRuntime, DType, Tensor, TypeConversionOps, VarBuilder, VarMap,
};
use oxidizr::config::{DTypePrecision, LoraConfig, OxidizrModelConfig};
use oxidizr::model::{
    load_checkpoint_into_model, save_checkpoint_from_params, FrozenParams, LitGPT,
};

/// A small dense-attention config: no `mamba_layers`, no `kv_latent_dim`
/// (so `Block::new` takes the `LegacyAttention` arm, not MLA), no
/// `num_experts` (so the FFN is a plain MLP, not MoE). Two layers is enough
/// to prove per-layer indices are named correctly while staying fast.
fn dense_cfg() -> OxidizrModelConfig {
    OxidizrModelConfig {
        name: "checkpoint-names-test".to_string(),
        hidden_size: 8,
        num_layers: 2,
        num_heads: 2,
        head_dim: None,
        kv_heads: None,
        kv_latent_dim: None,
        q_latent_dim: None,
        d_rope: None,
        num_experts: None,
        experts_per_tok: None,
        shared_expert_enabled: None,
        intermediate_size: Some(16),
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
        vocab_size: 16,
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

/// `dense_cfg`, with LoRA adapters on `q_proj` and `v_proj`. Proves the
/// `.lora_a` / `.lora_b` names `push_named_projection` emits are locked too.
fn lora_cfg() -> OxidizrModelConfig {
    let mut cfg = dense_cfg();
    cfg.name = "checkpoint-names-lora-test".to_string();
    cfg.lora = Some(LoraConfig {
        rank: 2,
        alpha: 4.0,
        target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        // The default freeze policy on purpose: freezing the whole base must
        // not drop a single name from the checkpoint, and this golden list is
        // what proves it.
        freeze_base: true,
        train_modules: Vec::new(),
    });
    cfg
}

/// `dense_cfg`, with per-head QK-norm on. The ONLY difference, so any name
/// that moves between the two golden lists moved because of this flag.
fn qk_norm_cfg() -> OxidizrModelConfig {
    let mut cfg = dense_cfg();
    cfg.name = "checkpoint-names-qk-norm-test".to_string();
    cfg.qk_norm = true;
    cfg
}

/// Build a model through the real public construction path: a fresh
/// `VarMap`/`VarBuilder`, then `LitGPT::new`, the same wiring
/// `commands::train` uses (minus the seed, which is irrelevant to names).
fn build(cfg: &OxidizrModelConfig) -> LitGPT<CpuRuntime> {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let mut varmap = VarMap::<CpuRuntime>::new();
    let mut vb = VarBuilder::new(&mut varmap, &device);
    LitGPT::<CpuRuntime>::new(cfg, &mut vb, &client).expect("dense model must build")
}

/// `model.named_parameters()`'s names, sorted.
fn sorted_names(model: &LitGPT<CpuRuntime>) -> Vec<String> {
    let mut names: Vec<String> = model
        .named_parameters()
        .into_iter()
        .map(|(name, _, _)| name)
        .collect();
    names.sort();
    names
}

/// Assert `model`'s parameter-name set equals `golden` exactly, reporting
/// which names are missing and which are unexpected rather than a bare
/// "left != right" on a multi-element vec.
fn assert_names_match(model: &LitGPT<CpuRuntime>, golden: &[&str], config_label: &str) {
    assert_name_set(&sorted_names(model), golden, config_label);
}

/// The set comparison itself, over any name list — used both for a model's
/// in-memory parameters and for the keys actually written to a checkpoint file.
fn assert_name_set(actual: &[String], golden: &[&str], config_label: &str) {
    let actual_set: BTreeSet<&str> = actual.iter().map(String::as_str).collect();
    let golden_set: BTreeSet<&str> = golden.iter().copied().collect();

    let missing: Vec<&str> = golden_set.difference(&actual_set).copied().collect();
    let unexpected: Vec<&str> = actual_set.difference(&golden_set).copied().collect();

    assert!(
        missing.is_empty() && unexpected.is_empty(),
        "{config_label}: parameter-name set diverged from the golden list.\n\
         missing (golden expects these, model does not have them): {missing:?}\n\
         unexpected (model has these, golden does not expect them): {unexpected:?}"
    );
}

/// `dense_cfg()`'s parameter names, sorted. Two layers x
/// (input_layernorm + q/k/v/o_proj + post_attention_layernorm +
/// gate/up/down_proj) plus embed_tokens/norm/lm_head (not tied).
const DENSE_GOLDEN: &[&str] = &[
    "embed_tokens.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.mlp.down_proj.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.v_proj.weight",
    "layers.1.input_layernorm.weight",
    "layers.1.mlp.down_proj.weight",
    "layers.1.mlp.gate_proj.weight",
    "layers.1.mlp.up_proj.weight",
    "layers.1.post_attention_layernorm.weight",
    "layers.1.self_attn.k_proj.weight",
    "layers.1.self_attn.o_proj.weight",
    "layers.1.self_attn.q_proj.weight",
    "layers.1.self_attn.v_proj.weight",
    "lm_head.weight",
    "norm.weight",
];

/// `qk_norm_cfg()`'s parameter names: `DENSE_GOLDEN` plus exactly
/// `self_attn.q_norm.weight` and `self_attn.k_norm.weight` per layer.
///
/// The names match HuggingFace's Qwen3 checkpoint
/// (`model.layers.N.self_attn.q_norm.weight`) minus the `model.` prefix
/// oxidizr does not use, so a Qwen3 import lands on them by name.
const QK_NORM_GOLDEN: &[&str] = &[
    "embed_tokens.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.mlp.down_proj.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.k_norm.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.self_attn.q_norm.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.v_proj.weight",
    "layers.1.input_layernorm.weight",
    "layers.1.mlp.down_proj.weight",
    "layers.1.mlp.gate_proj.weight",
    "layers.1.mlp.up_proj.weight",
    "layers.1.post_attention_layernorm.weight",
    "layers.1.self_attn.k_norm.weight",
    "layers.1.self_attn.k_proj.weight",
    "layers.1.self_attn.o_proj.weight",
    "layers.1.self_attn.q_norm.weight",
    "layers.1.self_attn.q_proj.weight",
    "layers.1.self_attn.v_proj.weight",
    "lm_head.weight",
    "norm.weight",
];

/// The two names `qk_norm` adds per layer, and the ONLY names it adds.
const QK_NORM_ADDED: &[&str] = &[
    "layers.0.self_attn.k_norm.weight",
    "layers.0.self_attn.q_norm.weight",
    "layers.1.self_attn.k_norm.weight",
    "layers.1.self_attn.q_norm.weight",
];

/// `lora_cfg()`'s parameter names: `DENSE_GOLDEN` plus `.lora_a`/`.lora_b`
/// for `q_proj` and `v_proj` in each layer (their `.weight` entries stay,
/// since the base is frozen, not dropped from `named_parameters`).
const LORA_GOLDEN: &[&str] = &[
    "embed_tokens.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.mlp.down_proj.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.self_attn.q_proj.lora_a",
    "layers.0.self_attn.q_proj.lora_b",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.v_proj.lora_a",
    "layers.0.self_attn.v_proj.lora_b",
    "layers.0.self_attn.v_proj.weight",
    "layers.1.input_layernorm.weight",
    "layers.1.mlp.down_proj.weight",
    "layers.1.mlp.gate_proj.weight",
    "layers.1.mlp.up_proj.weight",
    "layers.1.post_attention_layernorm.weight",
    "layers.1.self_attn.k_proj.weight",
    "layers.1.self_attn.o_proj.weight",
    "layers.1.self_attn.q_proj.lora_a",
    "layers.1.self_attn.q_proj.lora_b",
    "layers.1.self_attn.q_proj.weight",
    "layers.1.self_attn.v_proj.lora_a",
    "layers.1.self_attn.v_proj.lora_b",
    "layers.1.self_attn.v_proj.weight",
    "lm_head.weight",
    "norm.weight",
];

#[test]
fn dense_attention_config_has_the_golden_parameter_names() {
    let model = build(&dense_cfg());
    assert_names_match(&model, DENSE_GOLDEN, "dense config");
}

/// `qk_norm: true` must add `self_attn.q_norm.weight` /
/// `self_attn.k_norm.weight` per layer and NOTHING else.
#[test]
fn qk_norm_adds_exactly_q_norm_and_k_norm_per_layer() {
    let model = build(&qk_norm_cfg());
    assert_names_match(&model, QK_NORM_GOLDEN, "dense config with qk_norm");
}

/// The complement, stated as a diff rather than as two literal lists: the
/// `qk_norm` name set minus the `qk_norm: false` name set is exactly the two
/// norms per layer. This is what fails if a future change makes the flag drag
/// some other parameter along with it.
#[test]
fn qk_norm_changes_nothing_but_the_two_norms_per_layer() {
    let off: BTreeSet<String> = sorted_names(&build(&dense_cfg())).into_iter().collect();
    let on: BTreeSet<String> = sorted_names(&build(&qk_norm_cfg())).into_iter().collect();

    let added: Vec<&str> = on.difference(&off).map(String::as_str).collect();
    let removed: Vec<&str> = off.difference(&on).map(String::as_str).collect();

    assert_eq!(added, QK_NORM_ADDED, "qk_norm added the wrong names");
    assert!(
        removed.is_empty(),
        "qk_norm must not remove any parameter, but dropped: {removed:?}"
    );
}

#[test]
fn lora_on_q_and_v_proj_adds_exactly_the_adapter_names() {
    let model = build(&lora_cfg());
    assert_names_match(
        &model,
        LORA_GOLDEN,
        "dense config with LoRA on q_proj/v_proj",
    );
}

/// Freezing part of the base must not change which NAMES exist — only
/// which ones `named_trainable_parameters` reports as trainable.
/// `named_parameters` is documented as "every parameter a checkpoint must
/// carry, frozen ones included", so the golden list here is `DENSE_GOLDEN`
/// unchanged.
#[test]
fn freezing_part_of_the_base_does_not_change_the_name_set() {
    let model = build(&dense_cfg());

    let to_freeze = [
        "layers.0.self_attn.k_proj.weight",
        "layers.1.mlp.gate_proj.weight",
    ];
    let ids: Vec<TensorId> = model
        .named_parameters()
        .into_iter()
        .filter(|(name, _, _)| to_freeze.contains(&name.as_str()))
        .map(|(_, id, _)| id)
        .collect();
    assert_eq!(
        ids.len(),
        to_freeze.len(),
        "expected to find every name in `to_freeze` among the model's parameters"
    );

    let frozen_model = model
        .with_frozen_parameters(FrozenParams::from_ids(ids))
        .expect("rebuilding with a partially frozen base must succeed");

    assert_names_match(
        &frozen_model,
        DENSE_GOLDEN,
        "dense config with part of the base frozen",
    );
}

fn temp_checkpoint_path(tag: &str) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir()
        .join(format!(
            "oxidizr-checkpoint-names-{tag}-{}-{unique}.safetensors",
            std::process::id()
        ))
        .to_string_lossy()
        .into_owned()
}

/// A checkpoint saved from a dense-attention model must load back into a
/// freshly built model of the SAME config. This exercises
/// `save_checkpoint_from_params` and `load_checkpoint_into_model`
/// (`src/model.rs`) the same way `commands/train.rs` does on save/resume: it
/// is these two functions' name-keyed matching against
/// `extend_block_named_trainable`'s output that a renamed parameter breaks.
#[test]
fn dense_checkpoint_round_trips_through_save_and_load() {
    let cfg = dense_cfg();
    let model = build(&cfg);

    // Every parameter, not just the trainable ones: `save_checkpoint_from_params`
    // falls back to `var.tensor()` for a frozen entry absent from this map, but
    // nothing here is frozen, so handing it the full set (as
    // `commands/train.rs`'s optimizer-parameter map does for the trainable
    // subset) is the straightforward, correct input.
    let params: HashMap<TensorId, Tensor<CpuRuntime>> = model
        .named_parameters()
        .into_iter()
        .map(|(_, id, var)| (id, var.tensor().clone()))
        .collect();

    let path = temp_checkpoint_path("roundtrip");
    save_checkpoint_from_params(&path, &model, &params).expect("checkpoint save must succeed");

    // Assert the ON-DISK keys, not just that save->load agrees with itself.
    //
    // A round-trip alone cannot catch a rename: save and load run the same code,
    // so a renamed parameter is written and read back consistently and the
    // reload succeeds. Checking the keys actually present in the file is what
    // pins the checkpoint's external contract — the thing another tool, or a
    // future version of this one, has to agree with.
    let mut safetensors =
        boostr::format::SafeTensors::open(&path).expect("saved checkpoint must be readable");
    let on_disk = safetensors
        .load_all::<CpuRuntime>(&CpuDevice::new())
        .expect("saved checkpoint tensors must load");
    let mut disk_names: Vec<String> = on_disk.keys().cloned().collect();
    disk_names.sort();
    assert_name_set(&disk_names, DENSE_GOLDEN, "on-disk checkpoint keys");

    let mut reloaded = build(&cfg);
    let result = load_checkpoint_into_model(&mut reloaded, &path);
    let _ = std::fs::remove_file(&path);

    result.expect(
        "a checkpoint saved from this exact config must load back into a fresh model of the \
         same config without error",
    );
}

/// Every parameter of `model`, keyed by `TensorId` — the shape
/// `save_checkpoint_from_params` and `rebuild_with_parameters` both take.
fn parameter_map(model: &LitGPT<CpuRuntime>) -> HashMap<TensorId, Tensor<CpuRuntime>> {
    model
        .named_parameters()
        .into_iter()
        .map(|(_, id, var)| (id, var.tensor().clone()))
        .collect()
}

/// A `qk_norm: true` model must save and reload like any other: the on-disk
/// keys carry the two norms, and `load_checkpoint_into_model` accepts them.
#[test]
fn qk_norm_checkpoint_round_trips_through_save_and_load() {
    let cfg = qk_norm_cfg();
    let model = build(&cfg);
    let params = parameter_map(&model);

    let path = temp_checkpoint_path("qk-norm-roundtrip");
    save_checkpoint_from_params(&path, &model, &params).expect("checkpoint save must succeed");

    let mut safetensors =
        boostr::format::SafeTensors::open(&path).expect("saved checkpoint must be readable");
    let on_disk = safetensors
        .load_all::<CpuRuntime>(&CpuDevice::new())
        .expect("saved checkpoint tensors must load");
    let mut disk_names: Vec<String> = on_disk.keys().cloned().collect();
    disk_names.sort();
    assert_name_set(
        &disk_names,
        QK_NORM_GOLDEN,
        "on-disk qk_norm checkpoint keys",
    );

    let mut reloaded = build(&cfg);
    let result = load_checkpoint_into_model(&mut reloaded, &path);
    let _ = std::fs::remove_file(&path);

    result.expect("a qk_norm checkpoint must load back into a fresh model of the same config");
}

/// THE regression test for the silent-drop bug this unit closes.
///
/// A `qk_norm: true` model loading a checkpoint that lacks
/// `self_attn.q_norm.weight` / `self_attn.k_norm.weight` must hard-error and
/// NAME them.
///
/// Without the fix the two names are absent from `named_parameters()`, so
/// `load_checkpoint_into_model`'s missing-tensor check has nothing to miss:
/// the load "succeeds", the tensors are dropped on the floor, and the model
/// trains to fluent text while computing a different function from the one
/// the weights were trained for. This test is the assertion that the names
/// now exist and that the completeness check therefore covers them.
#[test]
fn a_qk_norm_model_rejects_a_checkpoint_without_the_qk_norm_tensors() {
    // Saved from a model WITHOUT qk_norm: every other name and shape matches,
    // so only the two norms are missing. Exactly the shape of a Qwen3 import
    // going the other way.
    let without = build(&dense_cfg());
    let params = parameter_map(&without);
    let path = temp_checkpoint_path("qk-norm-missing");
    save_checkpoint_from_params(&path, &without, &params).expect("checkpoint save must succeed");

    let mut with_qk_norm = build(&qk_norm_cfg());
    let result = load_checkpoint_into_model(&mut with_qk_norm, &path);
    let _ = std::fs::remove_file(&path);

    let err = result
        .expect_err(
            "loading a checkpoint that lacks q_norm/k_norm into a qk_norm model must fail: \
             silently random-initializing them is the exact bug this guards",
        )
        .to_string();
    for name in [
        "layers.0.self_attn.q_norm.weight",
        "layers.0.self_attn.k_norm.weight",
        "layers.1.self_attn.q_norm.weight",
        "layers.1.self_attn.k_norm.weight",
    ] {
        assert!(
            err.contains(name),
            "the error must name the missing tensor {name}, got: {err}"
        );
    }
}

/// The trainer rebuilds the model from the optimizer's parameter map once per
/// step. `q_norm`/`k_norm` must come back with the SAME `TensorId`s: a fresh
/// id detaches the AdamW state keyed by it, and the parameter silently stops
/// converging with nothing failing anywhere. This is the `qk_norm` half of
/// what `Trainer::assert_stable_parameter_ids` enforces at runtime.
#[test]
fn rebuilding_preserves_the_qk_norm_parameter_ids() {
    let model = build(&qk_norm_cfg());
    let before: Vec<(String, TensorId)> = model
        .named_parameters()
        .into_iter()
        .map(|(name, id, _)| (name, id))
        .collect();

    let rebuilt = model
        .rebuild_with_parameters(&parameter_map(&model))
        .expect("rebuilding a qk_norm model from its own parameters must succeed");
    let after: Vec<(String, TensorId)> = rebuilt
        .named_parameters()
        .into_iter()
        .map(|(name, id, _)| (name, id))
        .collect();

    assert_eq!(
        before, after,
        "every parameter must keep its name AND its TensorId across a rebuild"
    );

    // Not vacuous: the norms really are among the names checked above.
    let names: BTreeSet<&str> = after.iter().map(|(name, _)| name.as_str()).collect();
    for name in QK_NORM_ADDED {
        assert!(names.contains(name), "{name} must survive the rebuild");
    }
}

// ---------------------------------------------------------------------------
// Checkpoint dtype: the file must carry the model's OWN precision
// ---------------------------------------------------------------------------

/// `dense_cfg`, built at bf16. The ONLY difference, so anything that changes
/// between the two is caused by the precision.
fn bf16_cfg() -> OxidizrModelConfig {
    let mut cfg = dense_cfg();
    cfg.name = "checkpoint-names-bf16-test".to_string();
    cfg.dtype = DTypePrecision::BF16;
    cfg
}

/// A CPU tensor's values as `f32`, whatever dtype it holds.
///
/// Casting first is required: `to_vec::<f32>()` sizes its copy from `f32`, so
/// reading a bf16 tensor through it would run past the end of the storage.
fn f32_values(client: &CpuClient, tensor: &Tensor<CpuRuntime>) -> Vec<f32> {
    let tensor = tensor.contiguous().expect("contiguous");
    let tensor = if tensor.dtype() == DType::F32 {
        tensor
    } else {
        client
            .cast(&tensor, DType::F32)
            .expect("cast to f32")
            .contiguous()
            .expect("contiguous")
    };
    tensor.to_vec::<f32>()
}

/// Every parameter's `f32` values, keyed by name.
fn named_f32_values(model: &LitGPT<CpuRuntime>) -> Vec<(String, Vec<f32>)> {
    let client = CpuClient::new(CpuDevice::new());
    let mut out: Vec<(String, Vec<f32>)> = model
        .named_parameters()
        .into_iter()
        .map(|(name, _, var)| (name, f32_values(&client, var.tensor())))
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Save `model` and assert every tensor in the file is stored at `expected`,
/// at `expected`'s byte width and no wider.
///
/// The byte-width assertion is the one that matters. A values-only round-trip
/// cannot catch a re-widening to F32, because `load_tensors` casts each loaded
/// tensor to the config's dtype on the way in: the values come back correct
/// from a 4-byte-per-parameter file just as they do from a 2-byte one. Before
/// `stage_tensor_for_safetensors` kept the tensor's own dtype, this assertion
/// failed on the bf16 config with `F32` / 4 bytes per parameter — a checkpoint
/// twice the size it should be, which then materializes at full F32 width on
/// the device at load time.
fn assert_on_disk_dtype(model: &LitGPT<CpuRuntime>, expected: DType, tag: &str) {
    let path = temp_checkpoint_path(tag);
    save_checkpoint_from_params(&path, model, &parameter_map(model))
        .expect("checkpoint save must succeed");

    let safetensors =
        boostr::format::SafeTensors::open(&path).expect("saved checkpoint must be readable");
    let names: Vec<String> = safetensors.tensor_names().map(|s| s.to_string()).collect();
    assert!(
        !names.is_empty(),
        "the checkpoint must contain tensors for this assertion to mean anything"
    );

    for name in &names {
        let info = safetensors
            .tensor_info(name)
            .expect("a listed tensor must have info");
        assert_eq!(
            info.dtype, expected,
            "on-disk dtype of '{name}' must be the model's own dtype"
        );
        assert_eq!(
            info.size_bytes(),
            info.numel() * expected.size_in_bytes(),
            "'{name}' must occupy {} byte(s) per parameter on disk, not {}",
            expected.size_in_bytes(),
            info.size_bytes() / info.numel().max(1)
        );
    }

    drop(safetensors);
    let _ = std::fs::remove_file(&path);
}

/// An F32 model still writes an F32 checkpoint — the format that every
/// existing checkpoint and every tool reading them expects.
#[test]
fn an_f32_model_writes_an_f32_checkpoint() {
    assert_on_disk_dtype(&build(&dense_cfg()), DType::F32, "dtype-f32");
}

/// A bf16 model writes a bf16 checkpoint: 2 bytes per parameter, not 4.
#[test]
fn a_bf16_model_writes_a_bf16_checkpoint() {
    assert_on_disk_dtype(&build(&bf16_cfg()), DType::BF16, "dtype-bf16");
}

/// A bf16 checkpoint reloads into a bf16 model with every value intact. Saving
/// at the tensor's own width is lossless, so this is an EXACT comparison — no
/// cast happens in either direction.
#[test]
fn a_bf16_checkpoint_round_trips_without_losing_values() {
    let cfg = bf16_cfg();
    let model = build(&cfg);
    let before = named_f32_values(&model);

    let path = temp_checkpoint_path("bf16-roundtrip");
    save_checkpoint_from_params(&path, &model, &parameter_map(&model))
        .expect("checkpoint save must succeed");

    let mut reloaded = build(&cfg);
    let result = load_checkpoint_into_model(&mut reloaded, &path);
    let _ = std::fs::remove_file(&path);
    result.expect("a bf16 checkpoint must load back into a bf16 model");

    for (_, _, var) in reloaded.named_parameters() {
        assert_eq!(
            var.tensor().dtype(),
            DType::BF16,
            "a bf16 config must hold bf16 parameters after the load"
        );
    }
    assert_eq!(
        before,
        named_f32_values(&reloaded),
        "a bf16 round-trip must be exact"
    );
}

/// An F32 checkpoint — the only kind this trainer used to write — must still
/// load into a bf16-configured model, and a bf16 checkpoint must load into an
/// F32-configured one. `load_tensors` casts per tensor to the config's dtype
/// in both directions, so neither file is version-locked to the config that
/// wrote it.
#[test]
fn checkpoints_load_across_a_dtype_change_in_both_directions() {
    // Old-style F32 file -> bf16 model.
    let f32_model = build(&dense_cfg());
    let f32_path = temp_checkpoint_path("f32-into-bf16");
    save_checkpoint_from_params(&f32_path, &f32_model, &parameter_map(&f32_model))
        .expect("checkpoint save must succeed");

    let mut bf16_model = build(&bf16_cfg());
    let result = load_checkpoint_into_model(&mut bf16_model, &f32_path);
    let _ = std::fs::remove_file(&f32_path);
    result.expect("an F32 checkpoint must still load into a bf16-configured model");
    for (name, _, var) in bf16_model.named_parameters() {
        assert_eq!(
            var.tensor().dtype(),
            DType::BF16,
            "'{name}' must be cast down to the loading config's dtype"
        );
    }

    // New-style bf16 file -> F32 model.
    let source = build(&bf16_cfg());
    let bf16_path = temp_checkpoint_path("bf16-into-f32");
    save_checkpoint_from_params(&bf16_path, &source, &parameter_map(&source))
        .expect("checkpoint save must succeed");

    let mut f32_target = build(&dense_cfg());
    let result = load_checkpoint_into_model(&mut f32_target, &bf16_path);
    let _ = std::fs::remove_file(&bf16_path);
    result.expect("a bf16 checkpoint must load into an F32-configured model");
    for (name, _, var) in f32_target.named_parameters() {
        assert_eq!(
            var.tensor().dtype(),
            DType::F32,
            "'{name}' must be cast up to the loading config's dtype"
        );
    }

    // Widening bf16 -> F32 is exact, so the values must match the bf16 source.
    assert_eq!(
        named_f32_values(&source),
        named_f32_values(&f32_target),
        "widening a bf16 checkpoint to F32 must preserve every value"
    );
}
