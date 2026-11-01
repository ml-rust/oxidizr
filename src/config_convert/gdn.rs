//! Checks of `model.gdn_layers` and `model.gdn`, and the per-layer GDN
//! config `LitGPT` builds from.
//!
//! boostr's `ModelConfig` names one SSM kind per model in `hybrid_layers`,
//! so GDN layers stay out of it. `to_boostr` keeps `gdn: None`, and
//! `LitGPT` reads [`OxidizrModelConfig::gdn_config`] and
//! [`OxidizrModelConfig::is_gdn_layer`] instead.

use super::{checked_mamba_layers, model_error};
use crate::config::{GdnSpec, OxidizrModelConfig, ATTENTION_LORA_TARGETS, GDN_LORA_TARGETS};
use crate::prelude::Result;
use boostr::model::config::GdnConfig;

/// `model.gdn_layers`, sorted, after every check. Empty without GDN layers.
///
/// # Errors
///
/// A `ModelError` that names the key, when:
/// - `model.gdn` is set but `model.gdn_layers` is empty or absent.
/// - `model.gdn_layers` is non-empty but `model.gdn` is absent.
/// - An index is out of range or listed twice.
/// - An index is also in `model.mamba_layers`.
/// - The `model.gdn` shape fails a check of [`check_spec`].
pub(super) fn checked_gdn_layers(
    cfg: &OxidizrModelConfig,
    mamba_layers: &[usize],
) -> Result<Vec<usize>> {
    let layers = cfg.gdn_layers.as_deref().unwrap_or(&[]);
    if layers.is_empty() {
        if cfg.gdn.is_some() {
            return Err(model_error(
                "model.gdn is set but model.gdn_layers is empty, so no layer uses it. \
                 List the GDN layer indexes in model.gdn_layers, or remove model.gdn.",
            ));
        }
        return Ok(Vec::new());
    }
    let spec = cfg.gdn.as_ref().ok_or_else(|| {
        model_error(
            "model.gdn_layers is set but model.gdn is missing. Add a model.gdn block \
             with state_size, key_heads and value_heads.",
        )
    })?;
    if let Some(layer) = layers.iter().find(|layer| **layer >= cfg.num_layers) {
        return Err(model_error(format!(
            "model.gdn_layers contains out-of-range index {layer} (num_layers = {}).",
            cfg.num_layers
        )));
    }
    let mut sorted = layers.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() != layers.len() {
        return Err(model_error(format!(
            "model.gdn_layers contains duplicate indices: {layers:?}."
        )));
    }
    let shared: Vec<usize> = sorted
        .iter()
        .copied()
        .filter(|layer| mamba_layers.contains(layer))
        .collect();
    if !shared.is_empty() {
        return Err(model_error(format!(
            "model.gdn_layers and model.mamba_layers both name layers {shared:?}. A layer \
             is Mamba or GDN, never both. Remove each index from one of the two lists."
        )));
    }
    check_spec(spec, cfg.hidden_size)?;
    Ok(sorted)
}

/// Check the `model.gdn` shape before boostr sees it, so each error names
/// the YAML key.
fn check_spec(spec: &GdnSpec, hidden_size: usize) -> Result<()> {
    let counts = [
        ("state_size", spec.state_size),
        ("key_heads", spec.key_heads),
        ("value_heads", spec.value_heads),
        ("conv_kernel", spec.conv_kernel),
        ("chunk_size", spec.chunk_size),
    ];
    if let Some((name, _)) = counts.iter().find(|(_, value)| *value == 0) {
        return Err(model_error(format!("model.gdn.{name} must be > 0.")));
    }
    if !spec.value_heads.is_multiple_of(spec.key_heads) {
        return Err(model_error(format!(
            "model.gdn.value_heads ({}) must be a multiple of model.gdn.key_heads ({}). \
             Each key head serves value_heads / key_heads value heads.",
            spec.value_heads, spec.key_heads
        )));
    }
    if !(spec.rms_eps.is_finite() && spec.rms_eps > 0.0) {
        return Err(model_error(format!(
            "model.gdn.rms_eps ({}) must be finite and > 0.",
            spec.rms_eps
        )));
    }
    spec.to_boostr(hidden_size)
        .validate()
        .map_err(|e| model_error(format!("model.gdn: {e}")))
}

/// Refuse a key that no layer can honor once Mamba and GDN layers cover
/// every layer, and a LoRA target that no layer carries.
///
/// `mamba_layers` and `gdn_layers` come checked and disjoint.
///
/// # Errors
///
/// A `ModelError` that names the key and the fix.
pub(super) fn validate_gdn_roles(
    cfg: &OxidizrModelConfig,
    mamba_layers: &[usize],
    gdn_layers: &[usize],
) -> Result<()> {
    let has_attention = mamba_layers.len() + gdn_layers.len() < cfg.num_layers;
    if !has_attention && !gdn_layers.is_empty() {
        let extras = [
            ("rope_scaling", cfg.rope_scaling.is_some()),
            ("sliding_window", cfg.sliding_window.is_some()),
            ("qk_norm", cfg.qk_norm),
        ];
        if let Some((key, _)) = extras.iter().find(|(_, set)| *set) {
            return Err(model_error(format!(
                "{key} is set but model.mamba_layers and model.gdn_layers cover every \
                 layer, so no attention block reads it. Remove {key}, or leave at least \
                 one attention layer."
            )));
        }
    }
    let Some(lora) = &cfg.lora else {
        return Ok(());
    };
    for target in &lora.target_modules {
        let target = target.as_str();
        if GDN_LORA_TARGETS.contains(&target) && gdn_layers.is_empty() {
            return Err(model_error(format!(
                "lora.target_modules names {target:?}, a Gated DeltaNet projection, but \
                 model.gdn_layers is empty, so no layer carries it. Remove \
                 {target:?} from lora.target_modules, or add GDN layers."
            )));
        }
        if ATTENTION_LORA_TARGETS.contains(&target) && !has_attention {
            return Err(model_error(format!(
                "lora.target_modules names {target:?}, but model.mamba_layers and \
                 model.gdn_layers cover every layer, so no attention block carries it. \
                 Remove {target:?} from lora.target_modules, or leave at least one \
                 attention layer."
            )));
        }
    }
    Ok(())
}

impl OxidizrModelConfig {
    /// True when `model.gdn_layers` lists `layer_idx`.
    pub fn is_gdn_layer(&self, layer_idx: usize) -> bool {
        self.gdn_layers
            .as_ref()
            .is_some_and(|layers| layers.contains(&layer_idx))
    }

    /// True when `model.gdn_layers` lists at least one layer.
    pub fn has_gdn_layers(&self) -> bool {
        self.gdn_layers
            .as_ref()
            .is_some_and(|layers| !layers.is_empty())
    }

    /// boostr's `GdnConfig` of every GDN layer, or `None` without GDN
    /// layers.
    ///
    /// # Errors
    ///
    /// The errors of `mamba_layers` and `checked_gdn_layers`.
    pub fn gdn_config(&self) -> Result<Option<GdnConfig>> {
        let mamba_layers = checked_mamba_layers(self)?;
        if checked_gdn_layers(self, &mamba_layers)?.is_empty() {
            return Ok(None);
        }
        Ok(self
            .gdn
            .as_ref()
            .map(|spec| spec.to_boostr(self.hidden_size)))
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{GdnSpec, LitConfig, LoraConfig, OxidizrModelConfig};

    /// An 8-layer dense model with GDN layers 0 and 4.
    fn gdn_cfg() -> OxidizrModelConfig {
        let mut cfg = LitConfig::nano_baseline().model;
        cfg.gdn_layers = Some(vec![0, 4]);
        cfg.gdn = Some(GdnSpec {
            state_size: 64,
            key_heads: 4,
            value_heads: 8,
            conv_kernel: 4,
            chunk_size: 64,
            rms_eps: 1e-6,
        });
        cfg
    }

    fn lora(targets: &[&str]) -> LoraConfig {
        LoraConfig {
            rank: 2,
            alpha: 4.0,
            target_modules: targets.iter().map(|t| t.to_string()).collect(),
            freeze_base: true,
            train_modules: Vec::new(),
        }
    }

    fn error(cfg: &OxidizrModelConfig) -> String {
        cfg.to_boostr()
            .expect_err("this config must be refused")
            .to_string()
    }

    #[test]
    fn a_valid_gdn_hybrid_converts_and_reports_its_layers() {
        let cfg = gdn_cfg();
        let boostr = cfg.to_boostr().expect("valid GDN hybrid");
        assert!(
            boostr.gdn.is_none(),
            "GDN layers stay out of boostr's config"
        );
        assert!(cfg.is_gdn_layer(4) && !cfg.is_gdn_layer(1));
        let gdn = cfg.gdn_config().unwrap().expect("gdn config");
        assert_eq!(gdn.hidden_size, cfg.hidden_size);
        assert_eq!(gdn.inner_size, 8 * 64);
        assert!(LitConfig::nano_baseline()
            .model
            .gdn_config()
            .unwrap()
            .is_none());
    }

    #[test]
    fn an_out_of_range_index_is_refused() {
        let mut cfg = gdn_cfg();
        cfg.gdn_layers = Some(vec![0, 8]);
        let message = error(&cfg);
        assert!(
            message.contains("model.gdn_layers") && message.contains('8'),
            "{message}"
        );
    }

    #[test]
    fn a_duplicate_index_is_refused() {
        let mut cfg = gdn_cfg();
        cfg.gdn_layers = Some(vec![2, 2]);
        assert!(error(&cfg).contains("duplicate"));
    }

    #[test]
    fn an_index_shared_with_mamba_layers_is_refused() {
        let mut cfg = gdn_cfg();
        cfg.mamba_layers = Some(vec![1, 4]);
        cfg.mamba2_num_heads = Some(16);
        cfg.mamba2_head_dim = Some(64);
        cfg.mamba2_state_size = Some(16);
        cfg.mamba2_chunk_size = Some(16);
        let message = error(&cfg);
        assert!(
            message.contains("model.mamba_layers") && message.contains("[4]"),
            "{message}"
        );
    }

    #[test]
    fn gdn_layers_without_a_spec_is_refused() {
        let mut cfg = gdn_cfg();
        cfg.gdn = None;
        assert!(error(&cfg).contains("model.gdn is missing"));
    }

    #[test]
    fn a_spec_without_gdn_layers_is_refused() {
        let mut cfg = gdn_cfg();
        cfg.gdn_layers = None;
        assert!(error(&cfg).contains("model.gdn_layers is empty"));
    }

    #[test]
    fn indivisible_heads_are_refused() {
        let mut cfg = gdn_cfg();
        if let Some(spec) = cfg.gdn.as_mut() {
            spec.value_heads = 6;
        }
        let message = error(&cfg);
        assert!(message.contains("model.gdn.value_heads"), "{message}");
        assert!(message.contains("model.gdn.key_heads"), "{message}");
    }

    #[test]
    fn a_zero_count_is_refused_by_key() {
        let mut cfg = gdn_cfg();
        if let Some(spec) = cfg.gdn.as_mut() {
            spec.chunk_size = 0;
        }
        assert!(error(&cfg).contains("model.gdn.chunk_size"));
    }

    #[test]
    fn a_gdn_lora_target_needs_gdn_layers() {
        let mut cfg = LitConfig::nano_baseline().model;
        cfg.lora = Some(lora(&["in_proj_qkv"]));
        assert!(error(&cfg).contains("model.gdn_layers is empty"));

        let mut cfg = gdn_cfg();
        cfg.lora = Some(lora(&["in_proj_qkv", "out_proj", "q_proj"]));
        cfg.to_boostr().expect("GDN targets on a GDN hybrid");
    }

    #[test]
    fn a_gdn_lora_target_is_accepted_on_an_mla_hybrid() {
        let mut cfg = gdn_cfg();
        cfg.kv_heads = None;
        cfg.kv_latent_dim = Some(128);
        cfg.q_latent_dim = Some(128);
        cfg.d_rope = Some(16);
        cfg.lora = Some(lora(&["alpha_proj", "beta_proj", "gate_proj"]));
        cfg.to_boostr()
            .expect("GDN and MLP targets reach an MLA hybrid");
    }

    #[test]
    fn attention_keys_need_an_attention_layer() {
        let mut cfg = gdn_cfg();
        cfg.gdn_layers = Some((0..cfg.num_layers).collect());
        cfg.to_boostr().expect("a pure GDN model is valid");

        let mut with_lora = cfg.clone();
        with_lora.lora = Some(lora(&["q_proj"]));
        assert!(error(&with_lora).contains("no attention block carries it"));

        cfg.qk_norm = true;
        assert!(error(&cfg).contains("qk_norm"));
    }
}
