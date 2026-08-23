use super::*;
use crate::config::{LitConfig, RopeScalingSpec};

fn dense_baseline() -> OxidizrModelConfig {
    LitConfig::nano_baseline().model
}

fn hybrid_moe_baseline() -> OxidizrModelConfig {
    LitConfig::nano().model
}

#[test]
fn dense_attention_only() {
    let cfg = dense_baseline();
    let boostr_cfg = cfg.to_boostr().unwrap();
    boostr_cfg.validate().unwrap();

    assert_eq!(boostr_cfg.model_type, "llama");
    assert!(boostr_cfg.ssm.is_none());
    assert!(boostr_cfg.hybrid_layers.is_none());
    assert_eq!(boostr_cfg.intermediate_size, Some(2048));
}

#[test]
fn mla_dims_land_on_attention() {
    let cfg = OxidizrModelConfig {
        kv_heads: None,
        kv_latent_dim: Some(128),
        q_latent_dim: Some(128),
        d_rope: Some(16),
        ..dense_baseline()
    };
    let boostr_cfg = cfg.to_boostr().unwrap();
    boostr_cfg.validate().unwrap();

    let attn = boostr_cfg.attention.as_ref().unwrap();
    assert_eq!(attn.kv_latent_dim, Some(128));
    assert_eq!(attn.q_latent_dim, Some(128));
    assert_eq!(attn.d_rope, Some(16));
}

#[test]
fn moe_fields_carried_and_top_level_intermediate_size_cleared() {
    let cfg = hybrid_moe_baseline();
    let boostr_cfg = cfg.to_boostr().unwrap();
    boostr_cfg.validate().unwrap();

    let moe = boostr_cfg.moe.as_ref().unwrap();
    assert_eq!(moe.intermediate_size, Some(1536));
    assert!(boostr_cfg.intermediate_size.is_none());
}

#[test]
fn hybrid_mamba2_layers_are_exact_complements() {
    let cfg = hybrid_moe_baseline();
    let boostr_cfg = cfg.to_boostr().unwrap();
    boostr_cfg.validate().unwrap();

    let hybrid = boostr_cfg.hybrid_layers.as_ref().unwrap();
    let mut all_layers: Vec<usize> = hybrid
        .ssm_layers
        .iter()
        .chain(hybrid.attention_layers.iter())
        .copied()
        .collect();
    all_layers.sort_unstable();
    assert_eq!(all_layers, (0..cfg.num_layers).collect::<Vec<_>>());

    let ssm = boostr_cfg.ssm.as_ref().unwrap();
    assert_eq!(ssm.variant, "mamba2");
}

#[test]
fn mamba3_fields_carried() {
    let cfg = OxidizrModelConfig {
        mamba3_enabled: Some(true),
        mamba3_complex_rope: Some(true),
        mamba3_mimo_rank: Some(2),
        mamba3_use_conv: Some(false),
        ..hybrid_moe_baseline()
    };
    let boostr_cfg = cfg.to_boostr().unwrap();
    boostr_cfg.validate().unwrap();

    let ssm = boostr_cfg.ssm.as_ref().unwrap();
    assert_eq!(ssm.variant, "mamba3");
    assert_eq!(ssm.complex_rope, Some(true));
    assert_eq!(ssm.mimo_rank, Some(2));
    assert_eq!(ssm.use_conv, Some(false));
}

#[test]
fn all_ssm_layers_drop_attention_and_hybrid() {
    let base = dense_baseline();
    let cfg = OxidizrModelConfig {
        mamba_layers: Some((0..base.num_layers).collect()),
        mamba2_num_heads: Some(64),
        mamba2_head_dim: Some(16),
        mamba2_state_size: Some(64),
        mamba2_chunk_size: Some(64),
        mamba2_n_groups: Some(1),
        mamba2_conv_kernel: Some(4),
        mamba2_expand: Some(2),
        ..base
    };
    let boostr_cfg = cfg.to_boostr().unwrap();
    boostr_cfg.validate().unwrap();

    assert!(boostr_cfg.attention.is_none());
    assert!(boostr_cfg.hybrid_layers.is_none());
    assert_eq!(boostr_cfg.model_type, "mamba2");
}

#[test]
fn out_of_range_layer_index_is_rejected() {
    let base = dense_baseline();
    let cfg = OxidizrModelConfig {
        mamba_layers: Some(vec![base.num_layers]),
        ..base
    };
    assert!(cfg.to_boostr().is_err());
}

#[test]
fn duplicate_layer_index_is_rejected() {
    let cfg = OxidizrModelConfig {
        mamba_layers: Some(vec![0, 0]),
        ..dense_baseline()
    };
    assert!(cfg.to_boostr().is_err());
}

/// Bare Mamba1 predates Mamba2's head decomposition and has no `mamba2_*`
/// dims to give. It must convert to the `mamba1` variant rather than being
/// rejected for a field it never had — losing this silently turns a valid
/// Mamba1 config into a hard error.
#[test]
fn mamba_layers_without_mamba2_dims_is_mamba1() {
    let cfg = OxidizrModelConfig {
        mamba_layers: Some(vec![0]),
        mamba2_num_heads: None,
        ..dense_baseline()
    };
    let boostr_cfg = cfg.to_boostr().expect("bare mamba1 must convert");
    let ssm = boostr_cfg.ssm.as_ref().expect("ssm present");
    assert_eq!(ssm.variant, "mamba1");
    // One SISO channel spanning the whole inner width — this is also what
    // keeps `hidden_size * expand == num_heads * head_dim` satisfied.
    assert_eq!(ssm.num_heads, 1);
    assert_eq!(ssm.head_dim, cfg.hidden_size * ssm.expand);
    boostr_cfg.validate().expect("mamba1 config must validate");
}

/// Mamba3 genuinely needs the head dims, so their absence is still an error.
#[test]
fn mamba3_without_mamba2_dims_is_rejected() {
    let cfg = OxidizrModelConfig {
        mamba_layers: Some(vec![0]),
        mamba2_num_heads: None,
        mamba3_enabled: Some(true),
        ..dense_baseline()
    };
    assert!(cfg.to_boostr().is_err());
}

/// `tie_word_embeddings` must carry through to boostr's `ModelConfig`
/// unchanged in both directions — a tied checkpoint has no independent
/// `lm_head.weight`, so silently dropping this would break loading it.
#[test]
fn tie_word_embeddings_round_trips_both_ways() {
    let untied = OxidizrModelConfig {
        tie_word_embeddings: false,
        ..dense_baseline()
    };
    assert!(!untied.to_boostr().unwrap().tie_word_embeddings);

    let tied = OxidizrModelConfig {
        tie_word_embeddings: true,
        ..dense_baseline()
    };
    assert!(tied.to_boostr().unwrap().tie_word_embeddings);
}

fn yarn_scaling() -> RopeScalingSpec {
    RopeScalingSpec {
        scaling_type: "yarn".to_string(),
        factor: 4.0,
        original_max_position_embeddings: Some(4096),
        low_freq_factor: None,
        high_freq_factor: None,
        attention_factor: Some(1.0),
        beta_fast: Some(32.0),
        beta_slow: Some(1.0),
    }
}

/// A valid `AttentionOnly` config carries `sliding_window` and
/// `rope_scaling` straight through onto the resulting `AttentionConfig`.
#[test]
fn sliding_window_and_rope_scaling_land_on_attention() {
    let cfg = OxidizrModelConfig {
        sliding_window: Some(4096),
        rope_scaling: Some(yarn_scaling()),
        ..dense_baseline()
    };
    let boostr_cfg = cfg.to_boostr().unwrap();
    boostr_cfg.validate().unwrap();

    let attn = boostr_cfg.attention.as_ref().unwrap();
    assert_eq!(attn.sliding_window, Some(4096));
    let scaling = attn.rope_scaling.as_ref().unwrap();
    assert_eq!(scaling.scaling_type, "yarn");
    assert_eq!(scaling.factor, 4.0);
    assert_eq!(scaling.original_max_position_embeddings, Some(4096));
    assert_eq!(scaling.attention_factor, Some(1.0));
    assert_eq!(scaling.beta_fast, Some(32.0));
    assert_eq!(scaling.beta_slow, Some(1.0));
}

#[test]
fn sliding_window_on_hybrid_is_rejected() {
    let cfg = OxidizrModelConfig {
        sliding_window: Some(1024),
        ..hybrid_moe_baseline()
    };
    let err = cfg.to_boostr().unwrap_err().to_string();
    assert!(
        err.contains("sliding_window"),
        "error must name sliding_window: {err}"
    );
}

#[test]
fn attention_extras_on_ssm_only_are_rejected() {
    let base = dense_baseline();
    let ssm_only = OxidizrModelConfig {
        mamba_layers: Some((0..base.num_layers).collect()),
        mamba2_num_heads: Some(64),
        mamba2_head_dim: Some(16),
        mamba2_state_size: Some(64),
        mamba2_chunk_size: Some(64),
        mamba2_n_groups: Some(1),
        mamba2_conv_kernel: Some(4),
        mamba2_expand: Some(2),
        ..base
    };

    let with_rope_scaling = OxidizrModelConfig {
        rope_scaling: Some(yarn_scaling()),
        ..ssm_only.clone()
    };
    let err = with_rope_scaling.to_boostr().unwrap_err().to_string();
    assert!(
        err.contains("rope_scaling"),
        "error must name rope_scaling: {err}"
    );

    let with_sliding_window = OxidizrModelConfig {
        sliding_window: Some(1024),
        ..ssm_only.clone()
    };
    let err = with_sliding_window.to_boostr().unwrap_err().to_string();
    assert!(
        err.contains("sliding_window"),
        "error must name sliding_window: {err}"
    );

    let with_qk_norm = OxidizrModelConfig {
        qk_norm: true,
        ..ssm_only
    };
    let err = with_qk_norm.to_boostr().unwrap_err().to_string();
    assert!(err.contains("qk_norm"), "error must name qk_norm: {err}");
}

/// `qk_norm` builds `q_norm`/`k_norm` in the DENSE block only. The MLA block
/// normalizes its compressed latents instead and has nowhere to put them, so
/// asking for both must fail rather than drop the flag — the same silent-drop
/// class of bug the flag itself exists to close.
#[test]
fn qk_norm_on_an_mla_model_is_rejected() {
    let cfg = OxidizrModelConfig {
        qk_norm: true,
        kv_latent_dim: Some(128),
        q_latent_dim: Some(128),
        d_rope: Some(16),
        ..dense_baseline()
    };
    let err = cfg.to_boostr().unwrap_err().to_string();
    assert!(
        err.contains("qk_norm") && err.contains("kv_latent_dim"),
        "error must name qk_norm and the conflicting kv_latent_dim: {err}"
    );
}

/// The dense path accepts it: this is the config `LitGPT::new` then builds the
/// two norms from.
#[test]
fn qk_norm_on_a_dense_model_is_accepted() {
    let cfg = OxidizrModelConfig {
        qk_norm: true,
        ..dense_baseline()
    };
    cfg.to_boostr()
        .expect("qk_norm on an attention-only model must convert");
}

#[test]
fn unsupported_rope_scaling_type_is_rejected() {
    let cfg = OxidizrModelConfig {
        rope_scaling: Some(RopeScalingSpec {
            scaling_type: "dynamic".to_string(),
            ..yarn_scaling()
        }),
        ..dense_baseline()
    };
    let err = cfg.to_boostr().unwrap_err().to_string();
    assert!(
        err.contains("rope_scaling") && err.contains("dynamic"),
        "error must name rope_scaling and the offending value: {err}"
    );
}

#[test]
fn zero_sliding_window_is_rejected() {
    let cfg = OxidizrModelConfig {
        sliding_window: Some(0),
        ..dense_baseline()
    };
    let err = cfg.to_boostr().unwrap_err().to_string();
    assert!(
        err.contains("sliding_window"),
        "error must name sliding_window: {err}"
    );
}

/// A toy speech layout: text ids [0, 16), control ids [16, 32), audio ids
/// [32, 40), so `total_size() == 40`.
fn toy_speech_vocab() -> boostr::model::SpeechVocab {
    use boostr::model::{CodecVocab, SpeechVocab, ALL_SPECIAL_TOKENS};
    let codec = CodecVocab::new(2, 4).expect("valid codec");
    SpeechVocab::with_sequential_specials(16, 16, &ALL_SPECIAL_TOKENS, codec).expect("valid vocab")
}

#[test]
fn speech_vocab_matching_vocab_size_is_accepted() {
    let cfg = OxidizrModelConfig {
        vocab_size: 40,
        speech_vocab: Some(toy_speech_vocab()),
        ..dense_baseline()
    };
    cfg.to_boostr().unwrap();
}

#[test]
fn speech_vocab_total_size_mismatch_is_rejected() {
    let cfg = OxidizrModelConfig {
        vocab_size: 41,
        speech_vocab: Some(toy_speech_vocab()),
        ..dense_baseline()
    };
    let err = cfg.to_boostr().unwrap_err().to_string();
    assert!(err.contains("40"), "error must name total_size(): {err}");
    assert!(err.contains("41"), "error must name vocab_size: {err}");
}
