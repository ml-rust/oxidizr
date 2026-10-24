//! Tiny model configs for the recompute tests, one per block arm.
//!
//! Each config is `LitConfig::nano_baseline().model` cut down to 16 wide,
//! 8 tokens and a 12-token vocab, so a CPU forward and backward stay small.

use crate::config::{GdnSpec, LitConfig, LoraConfig, OxidizrModelConfig};
use crate::data::{BatchSegments, LitDataLoader};
use crate::model::LitGPT;
use crate::prelude::{Tensor, VarBuilder, VarMap};
use boostr::{CpuClient, CpuDevice, CpuRuntime};

/// The model type every fixture builds.
pub(in crate::model) type Model = LitGPT<CpuRuntime>;

/// Batch rows of [`token_ids`].
pub(in crate::model) const BATCH: usize = 2;
/// Tokens per row, equal to `max_seq_len`.
pub(in crate::model) const SEQ: usize = 8;
const VOCAB: usize = 12;

/// A two-layer dense GQA model with a SwiGLU MLP.
pub(in crate::model) fn legacy_mlp() -> OxidizrModelConfig {
    let mut cfg = LitConfig::nano_baseline().model;
    cfg.name = "recompute-test".to_string();
    cfg.hidden_size = 16;
    cfg.num_layers = 2;
    cfg.num_heads = 4;
    cfg.kv_heads = Some(2);
    cfg.intermediate_size = Some(32);
    cfg.vocab_size = VOCAB;
    cfg.tokenizer_vocab = "custom".to_string();
    cfg.max_seq_len = SEQ;
    cfg
}

/// Four routed experts, top-2, and a shared expert.
fn with_moe(cfg: OxidizrModelConfig) -> OxidizrModelConfig {
    OxidizrModelConfig {
        num_experts: Some(4),
        experts_per_tok: Some(2),
        shared_expert_enabled: Some(true),
        ..cfg
    }
}

/// [`legacy_mlp`] with an MoE FFN.
pub(in crate::model) fn legacy_moe() -> OxidizrModelConfig {
    with_moe(legacy_mlp())
}

/// [`legacy_mlp`] with MLA attention.
pub(in crate::model) fn mla_mlp() -> OxidizrModelConfig {
    OxidizrModelConfig {
        kv_heads: None,
        kv_latent_dim: Some(8),
        q_latent_dim: Some(8),
        d_rope: Some(2),
        ..legacy_mlp()
    }
}

/// [`mla_mlp`] with an MoE FFN.
pub(in crate::model) fn mla_moe() -> OxidizrModelConfig {
    with_moe(mla_mlp())
}

/// Every layer a bare Mamba1 block.
pub(in crate::model) fn mamba1() -> OxidizrModelConfig {
    OxidizrModelConfig {
        mamba_layers: Some(vec![0, 1]),
        mamba2_state_size: Some(4),
        mamba2_chunk_size: Some(4),
        ..legacy_mlp()
    }
}

/// Every layer a Mamba2 block. `16 * expand 2 == 4 heads * 8`.
pub(in crate::model) fn mamba2() -> OxidizrModelConfig {
    OxidizrModelConfig {
        mamba_layers: Some(vec![0, 1]),
        mamba2_num_heads: Some(4),
        mamba2_head_dim: Some(8),
        mamba2_state_size: Some(4),
        mamba2_chunk_size: Some(4),
        mamba2_n_groups: Some(1),
        mamba2_conv_kernel: Some(4),
        mamba2_expand: Some(2),
        ..legacy_mlp()
    }
}

/// [`mamba2`] with the Mamba3 mixer.
pub(in crate::model) fn mamba3() -> OxidizrModelConfig {
    OxidizrModelConfig {
        mamba3_enabled: Some(true),
        ..mamba2()
    }
}

/// GDN layer 0 and attention layer 1, both over an MoE FFN.
pub(in crate::model) fn gdn_moe() -> OxidizrModelConfig {
    with_moe(OxidizrModelConfig {
        gdn_layers: Some(vec![0]),
        gdn: Some(GdnSpec {
            state_size: 4,
            key_heads: 2,
            value_heads: 4,
            conv_kernel: 4,
            chunk_size: 4,
            rms_eps: 1e-6,
        }),
        ..legacy_mlp()
    })
}

/// [`legacy_mlp`] with LoRA on `q_proj` and `v_proj` over frozen bases.
pub(in crate::model) fn legacy_lora_frozen() -> OxidizrModelConfig {
    OxidizrModelConfig {
        lora: Some(LoraConfig {
            rank: 2,
            alpha: 4.0,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            freeze_base: true,
            train_modules: Vec::new(),
        }),
        ..legacy_mlp()
    }
}

/// The CPU device and a client on it.
pub(in crate::model) fn setup() -> (CpuDevice, CpuClient) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (device, client)
}

/// A seeded model of `cfg`.
pub(in crate::model) fn build(cfg: &OxidizrModelConfig) -> Model {
    let (device, client) = setup();
    let mut map = VarMap::<CpuRuntime>::new();
    let mut vb = VarBuilder::new(&mut map, &device).with_seed(0x5eed);
    LitGPT::new(cfg, &mut vb, &client).expect("the fixture model builds")
}

/// `[BATCH, SEQ]` token ids that cover the whole vocab.
pub(in crate::model) fn token_ids(device: &CpuDevice) -> Tensor<CpuRuntime> {
    let ids: Vec<i64> = (0..BATCH * SEQ)
        .map(|i| ((i * 5 + i / 3) % VOCAB) as i64)
        .collect();
    Tensor::<CpuRuntime>::from_slice(&ids, &[BATCH, SEQ], device).expect("token ids")
}

/// Document separator of [`segmented_batch`]. Input 4 of each row holds it.
const SEPARATOR: u32 = 9;

/// `[BATCH, SEQ]` input ids and their document segments, built by the data loader.
///
/// Each row splits into two documents after its separator.
pub(in crate::model) fn segmented_batch(
    device: &CpuDevice,
) -> (Tensor<CpuRuntime>, BatchSegments<CpuRuntime>) {
    let tokens: Vec<u32> = (0..BATCH * (SEQ + 1))
        .map(|i| ((i * 5 + i / 3) % VOCAB) as u32)
        .collect();
    let batch = LitDataLoader::<CpuRuntime>::new(tokens, BATCH, SEQ, device.clone())
        .with_document_mask(SEPARATOR)
        .next_batch()
        .expect("segmented batch");
    let segments = batch.segments.expect("the document mask is on");
    (batch.inputs, segments)
}
