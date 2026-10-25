//! An MLA checkpoint oxidizr writes serves through boostr with the logits oxidizr computes.
//!
//! Each test trains a tiny model a few steps and saves the final set and
//! `config.json` through oxidizr's own writers. boostr then loads the files as
//! blazr does: `config.json` as `UniversalConfig`, every tensor into a
//! `VarMap`, `restructure_hf_checkpoint`, then `LoadedModel::load`.
//!
//! - The config round-trips with every field intact.
//! - The served model reads every tensor the checkpoint holds.
//! - A one-shot prefill, a token-by-token decode and a mixed prefill-then-decode
//!   through the KV cache (and the SSM state of a hybrid) give oxidizr's logits.

mod common;

use boostr::format::{restructure_hf_checkpoint, SafeTensorsLoader};
use boostr::inference::{LayeredKvCache, LayeredKvCacheConfig, LayeredSsmState};
use boostr::model::{LoadedModel, UniversalConfig};
use boostr::{CpuClient, CpuDevice, CpuRuntime, DType, Tensor, VarBuilder, VarMap};
use common::{
    build_model, checkpoint_weights, data_loader, scratch_dir, tiny_config, train_parts, trainer,
};
use oxidizr::checkpoint_config::write_checkpoint_config;
use oxidizr::checkpoint_set::CheckpointLabel;
use oxidizr::config::{LitConfig, RopeScalingSpec};
use oxidizr::data::DataPosition;
use oxidizr::model::LitGPT;
use std::path::Path;

/// Optimizer steps before the save, so the served weights are trained ones.
const STEPS: usize = 3;
/// Context length of every model here, and the length of the compared sequence.
const SEQ: usize = 12;
/// Largest absolute logit difference the comparison accepts.
const TOL: f32 = 1e-4;

/// `tiny_config` with MLA attention on every layer: 2 heads of 8, latents 8 (KV) and 6 (Q), RoPE 4.
fn mla_config(tag: &str) -> LitConfig {
    let mut cfg = tiny_config(&scratch_dir(tag), STEPS);
    cfg.model.name = tag.to_string();
    cfg.model.num_heads = 2;
    cfg.model.kv_latent_dim = Some(8);
    cfg.model.q_latent_dim = Some(6);
    cfg.model.d_rope = Some(4);
    cfg.model.max_seq_len = SEQ;
    cfg
}

/// `cfg` with a MoE feed-forward of `experts` experts, `top_k` per token, and a shared expert.
fn with_moe(mut cfg: LitConfig, experts: usize, top_k: usize) -> LitConfig {
    cfg.model.num_experts = Some(experts);
    cfg.model.experts_per_tok = Some(top_k);
    cfg.model.intermediate_size = Some(12);
    cfg.trainer.load_balance_alpha = 0.01;
    cfg
}

/// Pure MLA with a dense SwiGLU MLP and tied embeddings.
fn mla_mlp_config() -> LitConfig {
    let mut cfg = mla_config("serve-mla-mlp");
    cfg.model.tie_word_embeddings = true;
    cfg
}

/// Pure MLA with a top-2 MoE of 3 experts plus a shared expert, and linear RoPE scaling.
fn mla_moe_config() -> LitConfig {
    let mut cfg = with_moe(mla_config("serve-mla-moe"), 3, 2);
    cfg.model.shared_expert_enabled = Some(true);
    cfg.model.rope_scaling = Some(RopeScalingSpec {
        scaling_type: "linear".to_string(),
        factor: 2.0,
        original_max_position_embeddings: None,
        low_freq_factor: None,
        high_freq_factor: None,
        attention_factor: None,
        beta_fast: None,
        beta_slow: None,
    });
    cfg
}

/// `models/nano-start/nano-start.yaml` scaled down: Mamba2 layers 0-2, then
/// one MLA layer with a top-1 MoE of 2 experts and the default shared expert.
/// `16 * 2 == 2 heads * 16`.
fn hybrid_config() -> LitConfig {
    let mut cfg = with_moe(mla_config("serve-hybrid"), 2, 1);
    cfg.model.num_layers = 4;
    cfg.model.mamba_layers = Some(vec![0, 1, 2]);
    cfg.model.mamba2_num_heads = Some(2);
    cfg.model.mamba2_head_dim = Some(16);
    cfg.model.mamba2_state_size = Some(4);
    cfg.model.mamba2_chunk_size = Some(4);
    cfg.model.mamba2_n_groups = Some(1);
    cfg.model.mamba2_conv_kernel = Some(4);
    cfg.model.mamba2_expand = Some(2);
    cfg
}

/// Trains `cfg` `STEPS` steps and saves the final set and `config.json` as a run does.
///
/// Returns the trained model and the weights file.
fn train_and_save(cfg: &LitConfig) -> (LitGPT<CpuRuntime>, String) {
    let model = build_model(cfg);
    let mut parts = train_parts(&model);
    let mut loader = data_loader(cfg);
    let layout = loader.row_layout().expect("row layout");
    loader.seek(DataPosition::default()).expect("seek");
    {
        let mut t = trainer(cfg, &mut parts, layout, 0, DataPosition::default());
        for _ in 0..STEPS {
            t.training_step(&model, &mut loader).expect("training step");
        }
        t.save_final_checkpoint(&model).expect("save the final set");
    }
    write_checkpoint_config(&cfg.model, &cfg.trainer.checkpoint_dir).expect("write config.json");
    let trained = model
        .rebuild_with_parameters(&parts.params)
        .expect("the trained model");
    (trained, checkpoint_weights(cfg, CheckpointLabel::Final))
}

/// A fixed token sequence over the whole context.
fn tokens(cfg: &LitConfig) -> Vec<i64> {
    let vocab = cfg.model.vocab_size as i64;
    (0..SEQ as i64).map(|i| (i * 7 + 3) % vocab).collect()
}

/// oxidizr's own logits `[1, SEQ, vocab]` over `ids`, flattened.
fn oxidizr_logits(model: &LitGPT<CpuRuntime>, ids: &[i64]) -> Vec<f32> {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let input = Tensor::<CpuRuntime>::from_slice(ids, &[1, ids.len()], &device).expect("input ids");
    let output = model.forward(&client, &input).expect("oxidizr forward");
    output
        .logits
        .tensor()
        .contiguous()
        .expect("contiguous")
        .to_vec::<f32>()
}

/// The `config.json` in `dir`, parsed as blazr parses an oxidizr checkpoint's.
fn read_config(dir: &Path) -> UniversalConfig {
    let json = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
    serde_json::from_str(&json).expect("config.json parses as UniversalConfig")
}

/// The written config equals the one `to_boostr` derives, field for field.
fn assert_config_round_trips(cfg: &LitConfig, written: &UniversalConfig) {
    let want = cfg.model.to_boostr().expect("to_boostr");
    assert_eq!(
        serde_json::to_value(written).expect("serialize the written config"),
        serde_json::to_value(&want).expect("serialize the derived config"),
        "config.json lost or changed a field"
    );
}

/// boostr's inference model over the weights at `weights`, loaded as blazr loads it.
///
/// Every tensor the file holds must reach the model.
fn served_model(config: &UniversalConfig, weights: &str) -> LoadedModel<CpuRuntime> {
    let device = CpuDevice::new();
    let mut loader = SafeTensorsLoader::open(weights).expect("open the weights file");
    let mut map = VarMap::<CpuRuntime>::new();
    for name in loader.tensor_names() {
        let tensor = loader
            .load_tensor::<CpuRuntime>(&name, &device)
            .unwrap_or_else(|e| panic!("load '{name}': {e}"));
        map.insert(name, tensor);
    }
    restructure_hf_checkpoint(&mut map, config, &device).expect("restructure");
    let model = {
        let mut vb = VarBuilder::new(&mut map, &device);
        LoadedModel::load(config, &mut vb).expect("boostr loads the oxidizr checkpoint")
    };
    let unread: Vec<&str> = map.names().collect();
    assert!(unread.is_empty(), "the served model never read {unread:?}");
    model
}

/// Logits of `model` over `ids`, fed as consecutive `chunks` through one cache.
fn served_logits(model: &LoadedModel<CpuRuntime>, ids: &[i64], chunks: &[usize]) -> Vec<f32> {
    let device = CpuDevice::new();
    let kv_config = LayeredKvCacheConfig {
        batch_size: 1,
        num_kv_heads: model.num_kv_heads().expect("KV heads"),
        initial_capacity: 4,
        max_seq_len: model.max_seq_len(),
        head_dim: model.head_dim().expect("KV head width"),
        dtype: DType::F32,
    };
    let attention_layers = model.num_attention_layers().expect("attention layers");
    let mut kv =
        LayeredKvCache::<CpuRuntime>::new(attention_layers, &kv_config, &device).expect("KV cache");
    let mut ssm = model.needs_ssm_state().then(|| {
        let layers = model.num_ssm_layers().expect("SSM layers");
        let mamba = model.mamba_config().expect("Mamba2 config");
        LayeredSsmState::<CpuRuntime>::new(layers, 1, mamba, DType::F32, &device)
            .expect("SSM state")
    });
    let mut logits = Vec::new();
    let mut at = 0;
    for &n in chunks {
        let piece =
            Tensor::<CpuRuntime>::from_slice(&ids[at..at + n], &[1, n], &device).expect("chunk");
        let out = match ssm.as_mut() {
            Some(state) => model.forward_hybrid(&piece, &mut kv, state, at),
            None => model.forward_with_kv_cache(&piece, &mut kv, at),
        }
        .unwrap_or_else(|e| panic!("served forward at position {at}: {e}"));
        logits.extend(out.contiguous().expect("contiguous").to_vec::<f32>());
        at += n;
    }
    assert_eq!(at, ids.len(), "the chunks cover the sequence");
    logits
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: logit count");
    let (at, diff) = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .enumerate()
        .fold(
            (0, 0.0f32),
            |best, (i, d)| if d > best.1 { (i, d) } else { best },
        );
    assert!(
        diff <= TOL,
        "{what}: logit {at} differs by {diff} (served {}, oxidizr {})",
        got[at],
        want[at]
    );
}

/// Trains and saves `cfg`, loads it through boostr, and compares every serving path with oxidizr.
fn assert_serves_like_oxidizr(cfg: &LitConfig, ssm_layers: usize) -> UniversalConfig {
    let (trained, weights) = train_and_save(cfg);
    let dir = Path::new(&cfg.trainer.checkpoint_dir);
    let config = read_config(dir);
    assert_config_round_trips(cfg, &config);

    let model = served_model(&config, &weights);
    assert!(
        matches!(model, LoadedModel::Hybrid(_)),
        "an MLA config serves as the hybrid model, got {model:?}"
    );
    assert_eq!(model.num_ssm_layers(), Some(ssm_layers));
    assert_eq!(
        model.num_attention_layers(),
        Some(cfg.model.num_layers - ssm_layers)
    );
    assert_eq!(model.needs_ssm_state(), ssm_layers > 0);

    let ids = tokens(cfg);
    let want = oxidizr_logits(&trained, &ids);
    assert_eq!(want.len(), SEQ * cfg.model.vocab_size);
    assert!(
        want.iter().all(|v| v.is_finite()),
        "non-finite oxidizr logits"
    );

    assert_close(&served_logits(&model, &ids, &[SEQ]), &want, "prefill");
    assert_close(
        &served_logits(&model, &ids, &[1; SEQ]),
        &want,
        "token-by-token decode",
    );
    assert_close(
        &served_logits(&model, &ids, &[5, 1, 3, 1, 2]),
        &want,
        "prefill then decode",
    );
    let _ = std::fs::remove_dir_all(dir);
    config
}

#[test]
fn an_mla_model_with_a_dense_mlp_serves_its_trained_logits() {
    let cfg = mla_mlp_config();
    let config = assert_serves_like_oxidizr(&cfg, 0);
    assert!(config.tie_word_embeddings);
    assert!(config.moe.is_none() && config.ssm.is_none() && config.hybrid_layers.is_none());
    assert_eq!(config.intermediate_size, Some(32));
    let attention = config.attention.expect("attention config");
    assert_eq!(
        (
            attention.kv_latent_dim,
            attention.q_latent_dim,
            attention.d_rope
        ),
        (Some(8), Some(6), Some(4))
    );
}

#[test]
fn an_mla_moe_model_serves_its_trained_logits() {
    let cfg = mla_moe_config();
    let config = assert_serves_like_oxidizr(&cfg, 0);
    let moe = config.moe.expect("moe config");
    assert_eq!((moe.num_experts, moe.experts_per_tok), (3, 2));
    assert_eq!(moe.shared_expert, Some(true));
    assert_eq!(moe.intermediate_size, Some(12));
    let scaling = config
        .attention
        .and_then(|a| a.rope_scaling)
        .expect("rope_scaling survives");
    assert_eq!(
        (scaling.scaling_type.as_str(), scaling.factor),
        ("linear", 2.0)
    );
}

#[test]
fn a_mamba2_mla_moe_hybrid_serves_its_trained_logits() {
    let cfg = hybrid_config();
    let config = assert_serves_like_oxidizr(&cfg, 3);
    assert_eq!(config.model_type, "hybrid");
    let split = config.hybrid_layers.expect("hybrid_layers");
    assert_eq!(
        (split.ssm_layers, split.attention_layers),
        (vec![0, 1, 2], vec![3])
    );
    let ssm = config.ssm.expect("ssm config");
    assert_eq!(
        (
            ssm.variant.as_str(),
            ssm.num_heads,
            ssm.head_dim,
            ssm.state_size,
            ssm.chunk_size,
            ssm.n_groups,
            ssm.conv_kernel,
            ssm.expand,
        ),
        ("mamba2", 2, 16, 4, 4, 1, 4, 2)
    );
    let moe = config.moe.expect("moe config");
    assert_eq!((moe.num_experts, moe.experts_per_tok), (2, 1));
    assert_eq!(moe.shared_expert, Some(true));
}
