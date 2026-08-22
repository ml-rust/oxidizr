use super::*;
use crate::config::LitConfig;

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

#[test]
fn missing_required_mamba2_field_is_rejected() {
    let cfg = OxidizrModelConfig {
        mamba_layers: Some(vec![0]),
        mamba2_num_heads: None,
        ..dense_baseline()
    };
    assert!(cfg.to_boostr().is_err());
}
