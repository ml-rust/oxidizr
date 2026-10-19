//! `trainer.document_mask`: attention and loss that stop at document boundaries.
//!
//! A separator token ends each document. With the key set, no token attends
//! across a separator, and no position predicts the first token of the next
//! document. This module holds the config, the meta record, and the model checks.

use super::{DTypePrecision, LitConfig};
use serde::{Deserialize, Serialize};
use std::fmt;

/// The head dims the CUDA varlen attention kernel takes.
pub const CUDA_VARLEN_HEAD_DIMS: [usize; 3] = [64, 128, 256];

/// YAML `trainer.document_mask`.
///
/// ```yaml
/// trainer:
///   document_mask: {}                # separator from the data sidecar
///   # document_mask: {separator_id: 2}
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DocumentMaskConfig {
    /// The id that ends every document. Absent takes `separator_id` from the
    /// `.tokens.json` sidecar of the data files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub separator_id: Option<u32>,
}

impl DocumentMaskConfig {
    /// The meta record of this config. `None` while `separator_id` is unresolved.
    pub fn record(&self) -> Option<DocumentMaskRecord> {
        self.separator_id
            .map(|separator_id| DocumentMaskRecord { separator_id })
    }
}

/// The document mask a checkpoint meta records: the resolved separator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentMaskRecord {
    /// The id that ends every document.
    pub separator_id: u32,
}

impl fmt::Display for DocumentMaskRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "separator_id {}", self.separator_id)
    }
}

/// `record` as resume messages print it: `off`, or its separator.
pub fn describe_document_mask(record: Option<DocumentMaskRecord>) -> String {
    record.map_or_else(|| "off".to_string(), |r| r.to_string())
}

/// Refuses `trainer.document_mask` on a model it cannot mask.
///
/// - A Mamba or GDN layer carries recurrent state across documents.
/// - `model.sliding_window`: varlen attention has no window.
/// - An MLA model needs `model.dtype: f32`: its document mask adds an F32 bias to the scores.
///
/// `--dtype` can change the dtype after this check, so the run plan calls it again.
///
/// # Errors
/// Returns an error naming the key and the layer that conflict.
pub fn check_document_mask_model(config: &LitConfig) -> anyhow::Result<()> {
    if config.trainer.document_mask.is_none() {
        return Ok(());
    }
    let model = &config.model;
    let listed = |layers: &Option<Vec<usize>>, i: usize| {
        layers.as_ref().is_some_and(|layers| layers.contains(&i))
    };
    for i in 0..model.num_layers {
        if listed(&model.mamba_layers, i) {
            anyhow::bail!(
                "trainer.document_mask cannot mask layer {i}: model.mamba_layers makes it a Mamba \
                 layer, and its recurrent state carries across documents. Remove \
                 trainer.document_mask, or remove {i} from model.mamba_layers."
            );
        }
        if listed(&model.gdn_layers, i) {
            anyhow::bail!(
                "trainer.document_mask cannot mask layer {i}: model.gdn_layers makes it a Gated \
                 DeltaNet layer, and its recurrent state carries across documents. Remove \
                 trainer.document_mask, or remove {i} from model.gdn_layers."
            );
        }
    }
    if let Some(window) = model.sliding_window.filter(|&w| w > 0) {
        anyhow::bail!(
            "trainer.document_mask cannot run with model.sliding_window {window}: varlen \
             attention has no window. Remove one of the two keys."
        );
    }
    if model.kv_latent_dim.is_some() && model.dtype != DTypePrecision::F32 {
        anyhow::bail!(
            "trainer.document_mask on an MLA model needs model.dtype f32, got {}: the document \
             mask adds an F32 bias to the attention scores. Set model.dtype: f32, or remove \
             trainer.document_mask.",
            model.dtype
        );
    }
    Ok(())
}

/// Refuses `trainer.document_mask` on a GQA model the CUDA varlen kernel cannot run.
///
/// The kernel takes a head_dim in [`CUDA_VARLEN_HEAD_DIMS`] and F32 or F16.
/// An MLA model runs its document mask through a bias, so this check skips it.
///
/// # Errors
/// Returns an error naming the head_dim or the dtype, with the accepted values.
pub fn check_document_mask_cuda(config: &LitConfig) -> anyhow::Result<()> {
    let model = &config.model;
    if config.trainer.document_mask.is_none() || model.kv_latent_dim.is_some() {
        return Ok(());
    }
    if model.num_heads == 0 {
        anyhow::bail!("model.num_heads must be non-zero");
    }
    let head_dim = model
        .head_dim
        .unwrap_or(model.hidden_size / model.num_heads);
    if !CUDA_VARLEN_HEAD_DIMS.contains(&head_dim) {
        anyhow::bail!(
            "trainer.document_mask on CUDA needs an attention head_dim in {CUDA_VARLEN_HEAD_DIMS:?}, \
             got {head_dim}. Set model.head_dim, or change model.hidden_size / model.num_heads, \
             or train on CPU."
        );
    }
    if model.dtype == DTypePrecision::BF16 {
        anyhow::bail!(
            "trainer.document_mask on CUDA needs model.dtype f32 or f16, got bf16: the varlen \
             attention kernel has no bf16 path. Set model.dtype: f16 or f32."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GdnSpec;

    /// A GQA model with head_dim 64 and the document mask on.
    fn masked() -> LitConfig {
        let mut cfg = LitConfig::nano_baseline();
        cfg.trainer.document_mask = Some(DocumentMaskConfig::default());
        cfg
    }

    fn model_error(cfg: &LitConfig) -> String {
        check_document_mask_model(cfg)
            .expect_err("the check must refuse this config")
            .to_string()
    }

    #[test]
    fn yaml_parses_with_and_without_a_separator() {
        let set: DocumentMaskConfig = serde_yaml::from_str("separator_id: 2").expect("parse");
        assert_eq!(set.record(), Some(DocumentMaskRecord { separator_id: 2 }));
        let unset: DocumentMaskConfig = serde_yaml::from_str("{}").expect("parse");
        assert_eq!(unset.record(), None);
        assert!(serde_yaml::from_str::<DocumentMaskConfig>("separator: 2").is_err());
    }

    #[test]
    fn an_attention_model_passes() {
        check_document_mask_model(&masked()).expect("GQA passes");
        check_document_mask_cuda(&masked()).expect("head_dim 64 passes on CUDA");
    }

    #[test]
    fn a_mamba_layer_is_refused_by_index() {
        let mut cfg = masked();
        cfg.model.mamba_layers = Some(vec![3]);
        let e = model_error(&cfg);
        assert!(
            e.contains("layer 3") && e.contains("model.mamba_layers"),
            "{e}"
        );
    }

    #[test]
    fn a_gdn_layer_is_refused_by_index() {
        let mut cfg = masked();
        cfg.model.gdn_layers = Some(vec![1]);
        cfg.model.gdn = Some(GdnSpec {
            state_size: 4,
            key_heads: 2,
            value_heads: 4,
            conv_kernel: 4,
            chunk_size: 4,
            rms_eps: 1e-6,
        });
        let e = model_error(&cfg);
        assert!(
            e.contains("layer 1") && e.contains("model.gdn_layers"),
            "{e}"
        );
    }

    #[test]
    fn a_sliding_window_is_refused_naming_both_keys() {
        let mut cfg = masked();
        cfg.model.sliding_window = Some(16);
        let e = model_error(&cfg);
        assert!(
            e.contains("model.sliding_window 16") && e.contains("trainer.document_mask"),
            "{e}"
        );
    }

    #[test]
    fn mla_needs_f32() {
        let mut cfg = masked();
        cfg.model.kv_latent_dim = Some(64);
        cfg.model.dtype = DTypePrecision::BF16;
        assert!(model_error(&cfg).contains("model.dtype f32"));
        cfg.model.dtype = DTypePrecision::F32;
        check_document_mask_model(&cfg).expect("f32 MLA passes");
    }

    #[test]
    fn cuda_refuses_an_unsupported_head_dim_with_its_value() {
        let mut cfg = masked();
        cfg.model.head_dim = Some(48);
        let e = check_document_mask_cuda(&cfg)
            .expect_err("head_dim 48")
            .to_string();
        assert!(e.contains("got 48") && e.contains("[64, 128, 256]"), "{e}");
    }

    #[test]
    fn cuda_refuses_bf16() {
        let mut cfg = masked();
        cfg.model.dtype = DTypePrecision::BF16;
        let e = check_document_mask_cuda(&cfg)
            .expect_err("bf16")
            .to_string();
        assert!(e.contains("bf16"), "{e}");
    }

    #[test]
    fn off_skips_every_check() {
        let mut cfg = LitConfig::nano_baseline();
        cfg.model.mamba_layers = Some(vec![0]);
        cfg.model.head_dim = Some(48);
        check_document_mask_model(&cfg).expect("off");
        check_document_mask_cuda(&cfg).expect("off");
    }
}
