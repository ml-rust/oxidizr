//! Conversion from oxidizr's YAML-facing [`OxidizrModelConfig`] to boostr's
//! [`ModelConfig`] (= `boostr::model::config::UniversalConfig`).
//!
//! boostr owns the canonical model-architecture schema; oxidizr's config is a
//! flatter, training-focused shape (Mamba2/3 fields spelled `mamba2_*`/`mamba3_*`,
//! hybrid layers as a single `mamba_layers` list, etc). This module is the one
//! place that bridges the two, so architecture features only need to be taught
//! to boostr once instead of hand-ported into every oxidizr consumer.

use crate::config::{OxidizrModelConfig, RopeScalingSpec};
use crate::prelude::Result;
use boostr::error::Error;
use boostr::model::config::{
    default_conv_kernel, default_expand, default_load_balance_alpha, default_n_groups,
    default_rms_norm_eps, default_z_loss_alpha, AttentionConfig, HybridConfig, ModelConfig,
    MoeConfig, RopeScalingConfig, SsmConfig,
};

/// `scaling_type` values boostr accepts (`AttentionConfig`/RoPE construction
/// hard-errors on anything else, deep inside model construction). Rejected
/// here instead, at config-conversion time.
const SUPPORTED_ROPE_SCALING_TYPES: [&str; 3] = ["linear", "llama3", "yarn"];

fn model_error(reason: impl Into<String>) -> Error {
    Error::ModelError {
        reason: reason.into(),
    }
}

/// Layer role derived from `mamba_layers` relative to `num_layers`.
enum LayerShape {
    /// No mamba layers: pure attention.
    AttentionOnly,
    /// Every layer is a mamba layer: pure SSM.
    SsmOnly,
    /// A non-empty strict subset of layers are mamba: hybrid.
    Hybrid,
}

fn layer_shape(mamba_layers: &[usize], num_layers: usize) -> LayerShape {
    if mamba_layers.is_empty() {
        LayerShape::AttentionOnly
    } else if mamba_layers.len() == num_layers {
        LayerShape::SsmOnly
    } else {
        LayerShape::Hybrid
    }
}

/// Validate `mamba_layers`: every index in range, no duplicates.
///
/// Both violations would silently corrupt the `HybridConfig` complement
/// (`attention_layers` = `0..num_layers` minus `mamba_layers`), so they must
/// be caught here rather than left to `HybridConfig::validate`.
fn checked_mamba_layers(cfg: &OxidizrModelConfig) -> Result<Vec<usize>> {
    let Some(layers) = cfg.mamba_layers.as_ref() else {
        return Ok(Vec::new());
    };
    for &layer in layers {
        if layer >= cfg.num_layers {
            return Err(model_error(format!(
                "mamba_layers contains out-of-range index {layer} (num_layers = {})",
                cfg.num_layers
            )));
        }
    }
    let mut sorted = layers.clone();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() != layers.len() {
        return Err(model_error(format!(
            "mamba_layers contains duplicate indices: {layers:?}"
        )));
    }
    Ok(sorted)
}

fn model_type(shape: &LayerShape, mamba3_enabled: bool) -> &'static str {
    match shape {
        LayerShape::AttentionOnly => "llama",
        LayerShape::Hybrid => "hybrid",
        LayerShape::SsmOnly => {
            if mamba3_enabled {
                "mamba3"
            } else {
                "mamba2"
            }
        }
    }
}

fn to_rope_scaling(spec: &RopeScalingSpec) -> RopeScalingConfig {
    RopeScalingConfig {
        scaling_type: spec.scaling_type.clone(),
        factor: spec.factor,
        original_max_position_embeddings: spec.original_max_position_embeddings,
        low_freq_factor: spec.low_freq_factor,
        high_freq_factor: spec.high_freq_factor,
        attention_factor: spec.attention_factor,
        beta_fast: spec.beta_fast,
        beta_slow: spec.beta_slow,
    }
}

/// Reject `rope_scaling`/`sliding_window` combinations boostr cannot honor.
/// Must run before conversion: boostr accepts these fields structurally on
/// every model type but only the `llama` (`AttentionOnly`) path actually
/// reads `sliding_window`, and only `llama`/`hybrid` read `rope_scaling` —
/// the rest would silently do nothing.
fn validate_attention_extras(cfg: &OxidizrModelConfig, shape: &LayerShape) -> Result<()> {
    if matches!(shape, LayerShape::SsmOnly) {
        if cfg.rope_scaling.is_some() {
            return Err(model_error(
                "rope_scaling is set but mamba_layers covers every layer (SsmOnly): \
                 there is no attention block to apply it to. Remove rope_scaling or \
                 leave at least one attention layer.",
            ));
        }
        if cfg.sliding_window.is_some() {
            return Err(model_error(
                "sliding_window is set but mamba_layers covers every layer (SsmOnly): \
                 there is no attention block to apply it to. Remove sliding_window or \
                 leave at least one attention layer.",
            ));
        }
        return Ok(());
    }

    if cfg.sliding_window.is_some() && matches!(shape, LayerShape::Hybrid) {
        return Err(model_error(
            "sliding_window is set on a hybrid model (mamba_layers is a non-empty \
             strict subset of layers): boostr's hybrid attention path never reads \
             sliding_window, so it would be silently ignored. Remove sliding_window or \
             make this model attention-only (mamba_layers empty).",
        ));
    }

    if let Some(scaling) = &cfg.rope_scaling {
        if !SUPPORTED_ROPE_SCALING_TYPES.contains(&scaling.scaling_type.as_str()) {
            return Err(model_error(format!(
                "rope_scaling.type is {:?}, which boostr does not support. Supported \
                 values are: {}.",
                scaling.scaling_type,
                SUPPORTED_ROPE_SCALING_TYPES.join(", ")
            )));
        }
    }

    if cfg.sliding_window == Some(0) {
        return Err(model_error(
            "sliding_window is set to 0: boostr treats 0 as the sentinel for \
             \"disabled\", so this reads as a window but silently means no window at \
             all. Remove sliding_window instead of setting it to 0.",
        ));
    }

    Ok(())
}

fn to_attention(cfg: &OxidizrModelConfig, shape: &LayerShape) -> Option<AttentionConfig> {
    if matches!(shape, LayerShape::SsmOnly) {
        return None;
    }
    Some(AttentionConfig {
        num_heads: cfg.num_heads,
        num_kv_heads: cfg.kv_heads,
        head_dim: None,
        rope_theta: cfg.rope_theta,
        rope_scaling: cfg.rope_scaling.as_ref().map(to_rope_scaling),
        kv_latent_dim: cfg.kv_latent_dim,
        q_latent_dim: cfg.q_latent_dim,
        d_rope: cfg.d_rope,
        sliding_window: cfg.sliding_window,
        // oxidizr's own attention block (src/model.rs) has no ALiBi path, and
        // boostr exposes no autograd/`var_*` ALiBi-biased attention primitive
        // for it to call — so this is always false, never a config knob.
        use_alibi: false,
    })
}

/// Required `SsmConfig` fields with no `Option` fallback in boostr. Missing
/// any of these when `mamba_layers` is non-empty is a hard error rather than
/// a silent default, since a wrong SSM shape corrupts training silently.
fn required_ssm_dims(cfg: &OxidizrModelConfig) -> Result<(usize, usize, usize, usize)> {
    let num_heads = cfg
        .mamba2_num_heads
        .ok_or_else(|| model_error("mamba_layers set but mamba2_num_heads is missing"))?;
    let head_dim = cfg
        .mamba2_head_dim
        .ok_or_else(|| model_error("mamba_layers set but mamba2_head_dim is missing"))?;
    let state_size = cfg
        .mamba2_state_size
        .ok_or_else(|| model_error("mamba_layers set but mamba2_state_size is missing"))?;
    let chunk_size = cfg
        .mamba2_chunk_size
        .ok_or_else(|| model_error("mamba_layers set but mamba2_chunk_size is missing"))?;
    Ok((num_heads, head_dim, state_size, chunk_size))
}

/// Mamba1 has no head decomposition — it is a single SISO channel of width
/// `d_inner`. boostr's `SsmConfig` requires `num_heads`/`head_dim` regardless of
/// variant, and validates `hidden_size * expand == num_heads * head_dim`, so the
/// faithful encoding is one head that spans the whole inner width. The classic
/// Mamba1 constants (state 16, conv kernel 4, expand 2, chunk 64) match
/// `boostr::model::Mamba1Config::new`, which is what actually gets built.
fn to_mamba1_ssm(cfg: &OxidizrModelConfig) -> SsmConfig {
    let expand = cfg.mamba2_expand.unwrap_or(2);
    SsmConfig {
        variant: "mamba1".to_string(),
        num_heads: 1,
        head_dim: cfg.hidden_size * expand,
        state_size: cfg.mamba2_state_size.unwrap_or(16),
        chunk_size: cfg.mamba2_chunk_size.unwrap_or(64),
        n_groups: cfg.mamba2_n_groups.unwrap_or_else(default_n_groups),
        conv_kernel: cfg.mamba2_conv_kernel.unwrap_or(4),
        expand,
        complex_rope: None,
        mimo_rank: None,
        use_conv: None,
    }
}

fn to_ssm(cfg: &OxidizrModelConfig, has_mamba: bool) -> Result<Option<SsmConfig>> {
    if !has_mamba {
        return Ok(None);
    }
    // No `mamba2_num_heads` means bare Mamba1, which predates the head-based
    // Mamba2 shape and cannot supply those dims. Encode it as such rather than
    // rejecting the config for a field Mamba1 never had.
    if cfg.mamba2_num_heads.is_none() && !cfg.mamba3_enabled.unwrap_or(false) {
        return Ok(Some(to_mamba1_ssm(cfg)));
    }
    let (num_heads, head_dim, state_size, chunk_size) = required_ssm_dims(cfg)?;
    let mamba3 = cfg.mamba3_enabled.unwrap_or(false);
    Ok(Some(SsmConfig {
        variant: if mamba3 { "mamba3" } else { "mamba2" }.to_string(),
        num_heads,
        head_dim,
        state_size,
        chunk_size,
        n_groups: cfg.mamba2_n_groups.unwrap_or_else(default_n_groups),
        conv_kernel: cfg.mamba2_conv_kernel.unwrap_or_else(default_conv_kernel),
        expand: cfg.mamba2_expand.unwrap_or_else(default_expand),
        complex_rope: cfg.mamba3_complex_rope,
        mimo_rank: cfg.mamba3_mimo_rank,
        use_conv: cfg.mamba3_use_conv,
    }))
}

fn to_moe(cfg: &OxidizrModelConfig) -> Result<Option<MoeConfig>> {
    let Some(num_experts) = cfg.num_experts else {
        return Ok(None);
    };
    let experts_per_tok = cfg
        .experts_per_tok
        .ok_or_else(|| model_error("num_experts set but experts_per_tok is missing"))?;
    Ok(Some(MoeConfig {
        num_experts,
        experts_per_tok,
        shared_expert: cfg.shared_expert_enabled,
        intermediate_size: cfg.intermediate_size,
        // Deliberately boostr's own defaults, NOT read from oxidizr: the real
        // load-balance/z-loss coefficients live on `TrainerConfig` and drive
        // the actual training loss there. Duplicating them onto the
        // architecture config would create a second source of truth that
        // silently drifts from what training actually uses. Called rather than
        // copied as literals so they cannot drift from boostr either.
        load_balance_alpha: default_load_balance_alpha(),
        z_loss_alpha: default_z_loss_alpha(),
    }))
}

fn to_hybrid_layers(
    shape: &LayerShape,
    ssm_layers: &[usize],
    num_layers: usize,
) -> Option<HybridConfig> {
    if !matches!(shape, LayerShape::Hybrid) {
        return None;
    }
    let attention_layers: Vec<usize> = (0..num_layers)
        .filter(|layer| !ssm_layers.contains(layer))
        .collect();
    Some(HybridConfig {
        ssm_layers: ssm_layers.to_vec(),
        attention_layers,
    })
}

impl OxidizrModelConfig {
    /// Convert this training-facing config into boostr's canonical
    /// [`ModelConfig`], failing loudly instead of substituting a default when
    /// a required boostr field has no oxidizr source.
    pub fn to_boostr(&self) -> Result<ModelConfig> {
        let ssm_layers = checked_mamba_layers(self)?;
        let shape = layer_shape(&ssm_layers, self.num_layers);
        let has_mamba = !ssm_layers.is_empty();
        let mamba3 = self.mamba3_enabled.unwrap_or(false);
        validate_attention_extras(self, &shape)?;

        let moe = to_moe(self)?;
        let intermediate_size = if moe.is_some() {
            None
        } else {
            self.intermediate_size
        };

        let config = ModelConfig {
            model_type: model_type(&shape, mamba3).to_string(),
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            max_seq_len: self.max_seq_len,
            intermediate_size,
            // oxidizr has no source field for this; take boostr's own default
            // rather than a copied literal so the two cannot drift.
            rms_norm_eps: default_rms_norm_eps(),
            attention: to_attention(self, &shape),
            ssm: to_ssm(self, has_mamba)?,
            moe,
            hybrid_layers: to_hybrid_layers(&shape, &ssm_layers, self.num_layers),
            tie_word_embeddings: self.tie_word_embeddings,
            grow_vocab: self.grow_vocab,
            // Not mapped: oxidizr-only fields (`name`, `tokenizer_vocab`, `dtype`)
            // have no boostr equivalent.
            vision: None,
            audio: None,
        };

        config.validate().map_err(|e| model_error(e.to_string()))?;
        Ok(config)
    }
}

#[cfg(test)]
#[path = "config_convert_tests.rs"]
mod tests;
