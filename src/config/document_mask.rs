//! `trainer.document_mask`: attention and loss that stop at document boundaries.
//!
//! A separator token ends each document. With the key set, no token attends
//! across a separator, and no position predicts the first token of the next
//! document. A Mamba or Gated DeltaNet layer resets its conv window and state at
//! each separator. This module holds the config, the meta record, and the CUDA check.

use super::LitConfig;
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

/// Whether any layer is attention: one in neither `mamba_layers` nor `gdn_layers`.
fn has_attention_layer(model: &super::OxidizrModelConfig) -> bool {
    let listed = |layers: &Option<Vec<usize>>, i: usize| {
        layers.as_ref().is_some_and(|layers| layers.contains(&i))
    };
    (0..model.num_layers).any(|i| !listed(&model.mamba_layers, i) && !listed(&model.gdn_layers, i))
}

/// Refuses `trainer.document_mask` on a GQA model the CUDA varlen kernel cannot run.
///
/// The kernel takes a head_dim in [`CUDA_VARLEN_HEAD_DIMS`] and F32, F16 and BF16.
/// An MLA model runs its document mask through a bias, so this check skips it.
/// A model with no attention layer has no varlen kernel to run, so it skips too.
///
/// # Errors
/// Returns an error naming the head_dim, with the accepted values.
pub fn check_document_mask_cuda(config: &LitConfig) -> anyhow::Result<()> {
    let model = &config.model;
    if config.trainer.document_mask.is_none()
        || model.kv_latent_dim.is_some()
        || !has_attention_layer(model)
    {
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DTypePrecision, GdnSpec};

    /// A GQA model with head_dim 64 and the document mask on.
    fn masked() -> LitConfig {
        let mut cfg = LitConfig::nano_baseline();
        cfg.trainer.document_mask = Some(DocumentMaskConfig::default());
        cfg
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
        check_document_mask_cuda(&masked()).expect("head_dim 64 passes on CUDA");
    }

    #[test]
    fn a_mamba_layer_is_accepted() {
        let mut cfg = masked();
        cfg.model.mamba_layers = Some(vec![3]);
        check_document_mask_cuda(&cfg).expect("Mamba layers take the mask");
    }

    #[test]
    fn a_hybrid_model_with_mamba_and_gdn_layers_is_accepted() {
        let mut cfg = masked();
        cfg.model.mamba_layers = Some(vec![0]);
        cfg.model.gdn_layers = Some(vec![1]);
        cfg.model.gdn = Some(GdnSpec {
            state_size: 4,
            key_heads: 2,
            value_heads: 4,
            conv_kernel: 4,
            chunk_size: 4,
            rms_eps: 1e-6,
        });
        check_document_mask_cuda(&cfg).expect("Mamba and GDN layers take the mask");
    }

    #[test]
    fn a_sliding_window_is_accepted() {
        let mut cfg = masked();
        cfg.model.sliding_window = Some(16);
        check_document_mask_cuda(&cfg).expect("the window runs inside each document");
    }

    #[test]
    fn an_mla_model_is_accepted_in_every_dtype() {
        let mut cfg = masked();
        cfg.model.kv_latent_dim = Some(64);
        for dtype in [
            DTypePrecision::F32,
            DTypePrecision::F16,
            DTypePrecision::BF16,
        ] {
            cfg.model.dtype = dtype;
            check_document_mask_cuda(&cfg).expect("MLA builds its mask bias in the model dtype");
        }
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
    fn cuda_skips_a_model_without_attention_layers() {
        let mut cfg = masked();
        cfg.model.head_dim = Some(48);
        cfg.model.dtype = DTypePrecision::BF16;
        cfg.model.mamba_layers = Some((0..cfg.model.num_layers).collect());
        check_document_mask_cuda(&cfg).expect("no attention layer, no varlen kernel");
    }

    #[test]
    fn cuda_accepts_every_dtype() {
        let mut cfg = masked();
        for dtype in [
            DTypePrecision::F32,
            DTypePrecision::F16,
            DTypePrecision::BF16,
        ] {
            cfg.model.dtype = dtype;
            check_document_mask_cuda(&cfg).expect("the varlen kernel runs F32, F16 and BF16");
        }
    }

    #[test]
    fn off_skips_every_check() {
        let mut cfg = LitConfig::nano_baseline();
        cfg.model.head_dim = Some(48);
        check_document_mask_cuda(&cfg).expect("off");
    }
}
