//! Document masks for the recurrent block arms.
//!
//! A packed row holds several documents. An attention arm reads the batch's
//! `BatchSegments`. A Mamba or Gated DeltaNet arm needs `SegmentMasks`: constants that
//! stop its conv window and its state at a document start. `LitGPT::forward_with`
//! builds them once per forward. Every recurrent layer shares them.

use crate::data::BatchSegments;
use crate::model::{Block, BlockType};
use crate::prelude::{DType, Result};
use boostr::nn::segment_masks::SegmentMasks;

impl<R> Block<R>
where
    R: boostr::Runtime<DType = DType>,
{
    /// The conv kernel of a recurrent arm. An attention arm has none.
    fn conv_kernel(&self) -> Option<usize> {
        match &self.block_type {
            BlockType::Attention { .. } | BlockType::LegacyAttention { .. } => None,
            BlockType::Mamba { block, .. } => Some(block.config().d_conv),
            BlockType::Mamba2 { block, .. } => Some(block.config().d_conv),
            BlockType::Mamba3 { block, .. } => Some(block.config().d_conv),
            BlockType::Gdn(layer) => Some(layer.block.config().conv_kernel),
        }
    }
}

/// The masks that `segments` implies for `layers`, in the compute `dtype`.
///
/// Returns `None` without segments, and for a model with no recurrent layer.
/// The masks cover the largest conv kernel of any layer.
///
/// # Errors
/// Returns an error when the segment ids are not an I32 `[batch, seq]` tensor.
pub(in crate::model) fn build_segment_masks<R, C>(
    client: &C,
    layers: &[Block<R>],
    segments: Option<&BatchSegments<R>>,
    dtype: DType,
) -> Result<Option<SegmentMasks<R>>>
where
    R: boostr::Runtime<DType = DType>,
    C: boostr::RuntimeClient<R>
        + boostr::CompareOps<R>
        + boostr::TypeConversionOps<R>
        + boostr::ShapeOps<R>,
{
    let Some(segments) = segments else {
        return Ok(None);
    };
    let Some(max_kernel) = layers.iter().filter_map(Block::conv_kernel).max() else {
        return Ok(None);
    };
    SegmentMasks::from_ids(client, segments.segment_ids(), max_kernel, dtype).map(Some)
}

#[cfg(test)]
mod tests {
    use crate::config::{ActivationCheckpointing, GdnSpec, OxidizrModelConfig};
    use crate::data::{BatchSegments, LitDataLoader};
    use crate::model::recompute::fixtures::{self, Model, SEQ};
    use crate::prelude::Tensor;
    use boostr::{CpuDevice, CpuRuntime};

    /// The id that ends a document. Input 3 of the packed row holds it.
    const SEP: u32 = 9;
    /// Packed row: document A is `[1, 2, 3, SEP]`, document B is `[5, 6, 7, 8]`.
    const PACKED: [u32; SEQ + 1] = [1, 2, 3, SEP, 5, 6, 7, 8, 4];
    const HALF: usize = SEQ / 2;
    const VOCAB: usize = 12;

    /// Attention layer 0, Mamba2 layer 1 and Gated DeltaNet layer 2.
    /// The Mamba conv kernel is 4 and the GDN kernel is 3.
    fn hybrid() -> OxidizrModelConfig {
        OxidizrModelConfig {
            num_layers: 3,
            mamba_layers: Some(vec![1]),
            gdn_layers: Some(vec![2]),
            gdn: Some(GdnSpec {
                state_size: 4,
                key_heads: 2,
                value_heads: 4,
                conv_kernel: 3,
                chunk_size: 4,
                rms_eps: 1e-6,
            }),
            ..fixtures::mamba2()
        }
    }

    fn ids(device: &CpuDevice, tokens: &[u32]) -> Tensor<CpuRuntime> {
        let ids: Vec<i64> = tokens.iter().map(|&t| i64::from(t)).collect();
        Tensor::<CpuRuntime>::from_slice(&ids, &[1, tokens.len()], device).expect("ids")
    }

    fn packed(device: &CpuDevice) -> (Tensor<CpuRuntime>, BatchSegments<CpuRuntime>) {
        let batch = LitDataLoader::<CpuRuntime>::new(PACKED.to_vec(), 1, SEQ, device.clone())
            .with_document_mask(SEP)
            .next_batch()
            .expect("packed batch");
        (
            batch.inputs,
            batch.segments.expect("the document mask is on"),
        )
    }

    fn logits(
        model: &Model,
        ids: &Tensor<CpuRuntime>,
        policy: &ActivationCheckpointing,
        segments: Option<&BatchSegments<CpuRuntime>>,
    ) -> Vec<f32> {
        let (_, client) = fixtures::setup();
        let out = model
            .forward_with(&client, ids, policy, segments)
            .expect("forward");
        out.logits
            .tensor()
            .contiguous()
            .expect("contiguous")
            .to_vec()
    }

    /// The packed row gives the logits of each document run as a row of its own.
    fn assert_packed_equals_separate(cfg: &OxidizrModelConfig, label: &str) {
        let (device, _) = fixtures::setup();
        let model = fixtures::build(cfg);
        let none = &ActivationCheckpointing::None;
        let (inputs, segments) = packed(&device);
        let both = logits(&model, &inputs, none, Some(&segments));
        let a = logits(&model, &ids(&device, &PACKED[..HALF]), none, None);
        let b = logits(&model, &ids(&device, &PACKED[HALF..SEQ]), none, None);
        let want: Vec<f32> = a.into_iter().chain(b).collect();
        assert_eq!(both.len(), want.len(), "{label}");
        let worst = both
            .iter()
            .zip(&want)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= 1e-5,
            "{label}: packed differs from rows by {worst}"
        );

        let crossing = logits(&model, &inputs, none, None);
        assert_ne!(crossing, both, "{label}: the segments changed nothing");
    }

    #[test]
    fn a_hybrid_packed_row_equals_its_documents_as_rows() {
        assert_packed_equals_separate(&hybrid(), "attention+mamba2+gdn");
    }

    #[test]
    fn a_mamba1_packed_row_equals_its_documents_as_rows() {
        assert_packed_equals_separate(&fixtures::mamba1(), "mamba1");
    }

    #[test]
    fn a_mamba3_packed_row_equals_its_documents_as_rows() {
        assert_packed_equals_separate(&fixtures::mamba3(), "mamba3");
    }

    #[test]
    fn a_gdn_packed_row_equals_its_documents_as_rows() {
        let cfg = OxidizrModelConfig {
            mamba_layers: None,
            ..hybrid()
        };
        assert_packed_equals_separate(&cfg, "attention+gdn");
    }

    #[test]
    fn checkpointed_equals_plain_under_segments() {
        let (device, _) = fixtures::setup();
        let (inputs, segments) = packed(&device);
        for (cfg, label) in [
            (hybrid(), "hybrid"),
            (fixtures::mamba1(), "mamba1"),
            (fixtures::mamba3(), "mamba3"),
        ] {
            let model = fixtures::build(&cfg);
            let plain = logits(
                &model,
                &inputs,
                &ActivationCheckpointing::None,
                Some(&segments),
            );
            let recomputed = logits(
                &model,
                &inputs,
                &ActivationCheckpointing::All,
                Some(&segments),
            );
            assert_eq!(plain, recomputed, "{label}");
        }
    }

    #[test]
    fn no_segments_keeps_a_recurrent_forward_unchanged() {
        let (device, client) = fixtures::setup();
        let model = fixtures::build(&hybrid());
        let (inputs, _) = packed(&device);
        let masks = super::build_segment_masks(&client, &model.layers, None, model.compute_dtype)
            .expect("no segments");
        assert!(masks.is_none());
        let a = logits(&model, &inputs, &ActivationCheckpointing::None, None);
        let b = logits(&model, &inputs, &ActivationCheckpointing::All, None);
        assert_eq!(a, b);
        assert_eq!(a.len(), SEQ * VOCAB);
    }

    #[test]
    fn the_masks_cover_the_largest_conv_kernel() {
        let (device, client) = fixtures::setup();
        let (_, segments) = packed(&device);
        let model = fixtures::build(&hybrid());
        let masks = super::build_segment_masks(
            &client,
            &model.layers,
            Some(&segments),
            model.compute_dtype,
        )
        .expect("masks")
        .expect("a recurrent layer needs masks");
        assert_eq!(masks.max_kernel(), 4);
        let attention = fixtures::build(&fixtures::legacy_mlp());
        let none = super::build_segment_masks(
            &client,
            &attention.layers,
            Some(&segments),
            attention.compute_dtype,
        )
        .expect("attention only");
        assert!(none.is_none());
    }
}
