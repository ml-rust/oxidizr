//! Conversion from oxidizr's YAML-facing [`OxidizrModelConfig`] to boostr's
//! [`ModelConfig`] (= `boostr::model::config::UniversalConfig`).
//!
//! boostr owns the canonical model-architecture schema; oxidizr's config is a
//! flatter, training-focused shape (Mamba2/3 fields spelled `mamba2_*`/`mamba3_*`,
//! hybrid layers as a single `mamba_layers` list, etc). This module is the one
//! place that bridges the two, so architecture features only need to be taught
//! to boostr once instead of hand-ported into every oxidizr consumer.

use crate::config::OxidizrModelConfig;
use crate::prelude::Result;
use boostr::error::Error;
use boostr::model::config::{
    default_conv_kernel, default_expand, default_load_balance_alpha, default_n_groups,
    default_rms_norm_eps, default_z_loss_alpha, AttentionConfig, HybridConfig, ModelConfig,
    MoeConfig, SsmConfig,
};

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

fn to_attention(cfg: &OxidizrModelConfig, shape: &LayerShape) -> Option<AttentionConfig> {
    if matches!(shape, LayerShape::SsmOnly) {
        return None;
    }
    Some(AttentionConfig {
        num_heads: cfg.num_heads,
        num_kv_heads: cfg.kv_heads,
        head_dim: None,
        rope_theta: cfg.rope_theta,
        rope_scaling: None,
        kv_latent_dim: cfg.kv_latent_dim,
        q_latent_dim: cfg.q_latent_dim,
        d_rope: cfg.d_rope,
        sliding_window: None,
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

fn to_ssm(cfg: &OxidizrModelConfig, has_mamba: bool) -> Result<Option<SsmConfig>> {
    if !has_mamba {
        return Ok(None);
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
    #[allow(dead_code)] // Call sites land in a later unit; conversion + tests only for now.
    pub fn to_boostr(&self) -> Result<ModelConfig> {
        let ssm_layers = checked_mamba_layers(self)?;
        let shape = layer_shape(&ssm_layers, self.num_layers);
        let has_mamba = !ssm_layers.is_empty();
        let mamba3 = self.mamba3_enabled.unwrap_or(false);

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
            // No oxidizr source yet; a later unit adds it.
            tie_word_embeddings: false,
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
