//! `model.gdn`: the shape of every Gated DeltaNet (GDN) layer.
//!
//! `model.gdn_layers` picks the layers that use it. The checks live in
//! `config_convert::gdn`.

use boostr::model::config::{
    default_gdn_chunk_size, default_gdn_conv_kernel, default_gdn_rms_eps, GdnConfig,
};
use serde::{Deserialize, Serialize};

/// Gated DeltaNet layer shape: boostr's `GdnConfig` minus the fields the
/// model fixes.
///
/// - `hidden_size` comes from `model.hidden_size`.
/// - `inner_size` is `value_heads * state_size`.
/// - `v_grouped` stays off. It marks a Hadamard-folded GGUF, never a trained model.
///
/// YAML:
///
/// ```yaml
/// gdn_layers: [0, 1, 2]
/// gdn:
///   state_size: 64
///   key_heads: 4
///   value_heads: 8
/// ```
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GdnSpec {
    /// Head width `S` of q, k and v. Each value head keeps an `[S, S]` state.
    pub state_size: usize,
    /// Query/key head count `H_k`.
    pub key_heads: usize,
    /// Value head count `H_v`. Must be a multiple of `key_heads`.
    pub value_heads: usize,
    /// Width of the depthwise causal conv over the qkv projection. Default 4.
    #[serde(default = "default_gdn_conv_kernel")]
    pub conv_kernel: usize,
    /// Chunk length of the chunked delta-rule recurrence. Default 64.
    #[serde(default = "default_gdn_chunk_size")]
    pub chunk_size: usize,
    /// Epsilon of the q/k L2 norm and the gated output norm. Default 1e-6.
    #[serde(default = "default_gdn_rms_eps")]
    pub rms_eps: f32,
}

impl GdnSpec {
    /// boostr's `GdnConfig` for a model of width `hidden_size`. The result
    /// is not checked. `config_convert::gdn` checks it.
    pub fn to_boostr(&self, hidden_size: usize) -> GdnConfig {
        GdnConfig {
            hidden_size,
            conv_kernel: self.conv_kernel,
            state_size: self.state_size,
            key_heads: self.key_heads,
            value_heads: self.value_heads,
            inner_size: self.value_heads * self.state_size,
            rms_eps: self.rms_eps,
            chunk_size: self.chunk_size,
            v_grouped: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{compute_model_config_hash, LitConfig};

    #[test]
    fn yaml_defaults_follow_boostr() {
        let spec: GdnSpec =
            serde_yaml::from_str("state_size: 4\nkey_heads: 2\nvalue_heads: 4\n").unwrap();
        assert_eq!(spec.conv_kernel, default_gdn_conv_kernel());
        assert_eq!(spec.chunk_size, default_gdn_chunk_size());
        assert_eq!(spec.rms_eps, default_gdn_rms_eps());
        let cfg = spec.to_boostr(8);
        cfg.validate().unwrap();
        assert_eq!(cfg.inner_size, 16);
        assert_eq!(cfg.hidden_size, 8);
        assert!(!cfg.v_grouped);
    }

    #[test]
    fn an_unknown_key_is_rejected() {
        let result: Result<GdnSpec, _> =
            serde_yaml::from_str("state_size: 4\nkey_heads: 2\nvalue_heads: 4\nnum_heads: 2\n");
        let message = result.expect_err("num_heads is no gdn key").to_string();
        assert!(message.contains("num_heads"), "{message}");
    }

    /// A config without GDN keeps its JSON, so every existing checkpoint
    /// keeps its config hash.
    #[test]
    fn absent_gdn_keys_do_not_change_the_config_hash() {
        let cfg = LitConfig::nano_baseline().model;
        assert!(cfg.gdn.is_none() && cfg.gdn_layers.is_none());
        let value = serde_json::to_value(&cfg).unwrap();
        let map = value.as_object().unwrap();
        assert!(!map.contains_key("gdn"), "gdn must be omitted when unset");
        assert!(
            !map.contains_key("gdn_layers"),
            "gdn_layers must be omitted when unset"
        );
        let mut parsed: crate::config::OxidizrModelConfig = serde_json::from_value(value).unwrap();
        assert_eq!(
            compute_model_config_hash(&parsed),
            compute_model_config_hash(&cfg)
        );

        parsed.gdn_layers = Some(vec![0]);
        parsed.gdn = Some(GdnSpec {
            state_size: 64,
            key_heads: 4,
            value_heads: 8,
            conv_kernel: 4,
            chunk_size: 64,
            rms_eps: 1e-6,
        });
        assert_ne!(
            compute_model_config_hash(&parsed),
            compute_model_config_hash(&cfg),
            "GDN layers change the parameter set, so they must change the hash"
        );
    }
}
