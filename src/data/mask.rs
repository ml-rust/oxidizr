//! Loss-mask rules and the mask builder shared by both loaders.

use boostr::model::SpeechLayout;

/// Which target ids get scored by the loss.
///
/// One facility for every masked-loss scheme, so a second consumer never
/// forks a parallel copy of the mask builder. Both loaders
/// (`LitDataLoader::next_batch` and `PrefetchDataLoader::prefetch_worker`)
/// build their mask through [`build_target_mask`], so the two shift paths
/// cannot diverge.
pub enum TargetMask {
    /// boostr's speech-LM id layout: a target scores iff it falls in the
    /// layout's audio region ([`SpeechLayout::is_audio`]).
    SpeechAudio(SpeechLayout),
    /// Prefix-LM continuation scoring: a target scores iff its id is at
    /// least `target_base` (the continuation region — e.g. phoneme ids in a
    /// G2P vocab laid out as `specials | graphemes | phonemes`), or it
    /// appears in `always_score`.
    ///
    /// `always_score` names ids the model must still learn to emit that sit
    /// BELOW `target_base` — most notably `<eos>`, which lives in the
    /// specials region but ends the continuation and must be scored like any
    /// other continuation token. Without it, `<eos>` never contributes to the
    /// loss and the model never learns to stop.
    PrefixLm {
        /// First id of the continuation region. Every id `>= target_base` scores.
        target_base: u32,
        /// Ids below `target_base` that score regardless (typically `[eos_id]`).
        always_score: Vec<u32>,
    },
}

impl TargetMask {
    /// Whether a target id at this position scores in the loss.
    fn is_scored(&self, target: u32) -> bool {
        match self {
            Self::SpeechAudio(layout) => layout.is_audio(target),
            Self::PrefixLm {
                target_base,
                always_score,
            } => target >= *target_base || always_score.contains(&target),
        }
    }
}

/// Build the loss mask over a raw `[batch_size, seq_len + 1]` token window.
///
/// The mask is derived from the POST-SHIFT targets (`window[row_start + s + 1]`),
/// never from the inputs (`window[row_start + s]`). Masking on inputs asks
/// "is the current token scored" instead of "is the token being predicted
/// scored", shifting every masked-in span by exactly one position on every row.
///
/// Each window row is one fixed row of the stream (see [`super::rows`]). A
/// row can hold several packed examples back to back with no
/// document-boundary marker, so the mask is per-token, never a single
/// boundary index per row. A row can also start mid-sequence when the data
/// was not packed to row boundaries; that is harmless here because scoring
/// depends only on each target id's own value (`is_scored`), never on its
/// position within a sequence. Without `trainer.document_mask`, attention
/// crosses sequence boundaries inside a row, as in standard packed LM
/// training. [`super::host_masks`] adds the document mask.
///
/// Both loaders call it through `LitDataLoader::host_masks`, so the two
/// shift paths cannot diverge.
pub(super) fn build_target_mask(
    mask: &TargetMask,
    window: &[u32],
    batch_size: usize,
    seq_len: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(batch_size * seq_len);
    for b in 0..batch_size {
        let row_start = b * (seq_len + 1);
        for s in 0..seq_len {
            let target = window[row_start + s + 1];
            out.push(if mask.is_scored(target) { 1.0 } else { 0.0 });
        }
    }
    out
}

/// Number of positions a mask from [`build_target_mask`] scores.
///
/// The mask holds exactly `0.0` and `1.0`, so this counts the `1.0` entries.
pub(super) fn scored_count(mask: &[f32]) -> usize {
    mask.iter().filter(|m| **m != 0.0).count()
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::data::{Batch, LitDataLoader};
    use crate::prelude::Tensor;
    use boostr::model::{CodecVocab, SpeechVocab, ALL_SPECIAL_TOKENS};
    use boostr::{CpuDevice, CpuRuntime};

    const TEXT_VOCAB: usize = 16;
    /// Must hold all 13 control tokens.
    const CONTROL_REGION: usize = 16;

    /// Toy layout: text ids [0, 16), control ids [16, 32), audio ids [32, 40).
    fn toy_vocab() -> SpeechVocab {
        let codec = CodecVocab::new(2, 4).expect("valid codec");
        SpeechVocab::with_sequential_specials(
            TEXT_VOCAB,
            CONTROL_REGION,
            &ALL_SPECIAL_TOKENS,
            codec,
        )
        .expect("valid vocab")
    }

    pub(in crate::data) fn audio_base() -> u32 {
        (TEXT_VOCAB + CONTROL_REGION) as u32
    }

    pub(in crate::data) fn mask_of(batch: &Batch<CpuRuntime>) -> Vec<f32> {
        batch
            .loss_mask
            .as_ref()
            .expect("loss mask present")
            .contiguous()
            .expect("contiguous mask")
            .to_vec()
    }

    pub(in crate::data) fn ids(t: &Tensor<CpuRuntime>) -> Vec<i64> {
        t.contiguous().expect("contiguous").to_vec()
    }

    pub(in crate::data) fn loader(
        tokens: Vec<u32>,
        batch_size: usize,
        seq_len: usize,
    ) -> LitDataLoader<CpuRuntime> {
        LitDataLoader::<CpuRuntime>::new(tokens, batch_size, seq_len, CpuDevice::new())
            .with_speech_layout(SpeechLayout::Native(toy_vocab()))
    }

    /// The mask must answer "is the token being PREDICTED audio", so it is built
    /// from the post-shift targets. Building it from `inputs` instead would move
    /// the first `1.0` one position later, which is what this asserts against.
    #[test]
    fn mask_is_derived_from_targets_not_inputs() {
        let base = audio_base();
        // inputs  = [0, 1, 2, base+0, base+1]
        // targets = [1, 2, base+0, base+1, base+2]
        let tokens = vec![0, 1, 2, base, base + 1, base + 2];
        let batch = loader(tokens, 1, 5).next_batch().expect("batch");

        let mask = mask_of(&batch);
        assert_eq!(mask, vec![0.0, 0.0, 1.0, 1.0, 1.0]);

        let first_one = mask
            .iter()
            .position(|v| *v == 1.0)
            .expect("a masked-in position");
        assert_eq!(
            first_one, 2,
            "mask must start where the TARGET is the first audio token (index 2), \
             not where the INPUT is (index 3)"
        );

        // Sanity: the target at that index really is the first audio token.
        assert_eq!(ids(&batch.targets)[first_one], base as i64);
    }

    /// GPT-style packing puts several text->audio examples in one row, so the
    /// mask must be per-token and produce one span per example.
    #[test]
    fn packed_row_with_two_segments_produces_two_mask_spans() {
        let base = audio_base();
        // targets = [1, base, base, 3, 4, base, base, 5]
        let tokens = vec![0, 1, base, base, 3, 4, base, base, 5];
        let batch = loader(tokens, 1, 8).next_batch().expect("batch");

        assert_eq!(
            mask_of(&batch),
            vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
        );
        assert_eq!(batch.scored, 4, "scored counts the host mask's 1.0 entries");
    }

    #[test]
    fn no_speech_layout_means_no_loss_mask() {
        let base = audio_base();
        let tokens = vec![0, 1, base, base + 1, 2, 3];
        let mut l = LitDataLoader::<CpuRuntime>::new(tokens, 1, 5, CpuDevice::new());
        let batch = l.next_batch().expect("batch");
        assert!(batch.loss_mask.is_none());
        assert_eq!(batch.scored, 5, "an unmasked batch scores every position");
        assert_eq!(batch.inputs.shape(), [1usize, 5]);
        assert_eq!(batch.targets.shape(), [1usize, 5]);
    }

    // ------------------------------------------------------------------
    // Prefix-LM continuation masking (`TargetMask::PrefixLm`)
    // ------------------------------------------------------------------
    //
    // Toy G2P vocab: specials [0, 4) = {bos=0, dialect=1, sep=2, eos=3},
    // graphemes [4, 12), phonemes [12, 20). `target_base = 12`, so a target
    // scores iff it is a phoneme id or is `eos` (3), which sits below
    // `target_base` in the specials region but still ends the continuation.

    pub(in crate::data) const EOS: u32 = 3;
    pub(in crate::data) const PREFIX_LM_TARGET_BASE: u32 = 12;

    pub(in crate::data) fn prefix_lm_loader(
        tokens: Vec<u32>,
        batch_size: usize,
        seq_len: usize,
    ) -> LitDataLoader<CpuRuntime> {
        LitDataLoader::<CpuRuntime>::new(tokens, batch_size, seq_len, CpuDevice::new())
            .with_target_mask(TargetMask::PrefixLm {
                target_base: PREFIX_LM_TARGET_BASE,
                always_score: vec![EOS],
            })
    }

    /// A full `<bos> <dialect> w o r d <sep> p h o n e m e s <eos>` sequence:
    /// prompt tokens (bos/dialect/graphemes/sep) never score, phoneme targets
    /// always score, and the closing `<eos>` target scores despite being a
    /// specials-region id below `target_base`.
    #[test]
    fn prefix_lm_scores_phonemes_and_eos_never_prompt() {
        // bos, dialect, w,o,r,d (graphemes), sep, p,h,o,n,e,m,s (phonemes), eos
        let tokens = vec![0, 1, 4, 5, 6, 7, 2, 12, 13, 14, 15, 16, 17, 18, EOS];
        let batch = prefix_lm_loader(tokens, 1, 14).next_batch().expect("batch");

        assert_eq!(
            mask_of(&batch),
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "targets: dialect,w,o,r,d,sep (unscored) then p,h,o,n,e,m,s,eos (scored)"
        );
    }

    /// `target_base` is the first scored id: one below it never scores, the
    /// id itself always does.
    #[test]
    fn prefix_lm_target_base_boundary_is_inclusive() {
        let below = PREFIX_LM_TARGET_BASE - 1; // last grapheme id
        let at_base = PREFIX_LM_TARGET_BASE; // first phoneme id
        let tokens = vec![0, below, at_base, at_base + 1];
        let batch = prefix_lm_loader(tokens, 1, 3).next_batch().expect("batch");
        // targets = [below, at_base, at_base+1]
        assert_eq!(mask_of(&batch), vec![0.0, 1.0, 1.0]);
    }

    /// A row can start mid-sequence when `<bos>...<eos>` examples are packed
    /// back to back without regard to row boundaries. Scoring depends only on
    /// each target id's own value, so a row that begins inside the first
    /// example's continuation still masks every position correctly.
    #[test]
    fn row_starting_mid_sequence_still_masks_by_id() {
        // seq A: bos, dialect, w(grapheme), sep, p,p(phonemes), eos     (7 tokens)
        // seq B: bos, dialect, o,r(graphemes), sep, p,h(phonemes), eos  (8 tokens)
        let tokens = vec![0, 1, 4, 2, 12, 13, EOS, 0, 1, 5, 6, 2, 13, 14, EOS];
        let mut loader = prefix_lm_loader(tokens, 1, 4);

        // Row 0 = tokens[0..5] = [bos, dialect, w, sep, p]; starts at a
        // sequence boundary.
        let first = loader.next_batch().expect("first batch");
        assert_eq!(mask_of(&first), vec![0.0, 0.0, 0.0, 1.0]);

        // Row 1 = tokens[5..10] = [p, eos, bos, dialect, o]: starts at seq
        // A's last phoneme, not at a `<bos>`.
        let second = loader.next_batch().expect("second batch");
        // targets = [eos, bos, dialect, o(grapheme)]
        assert_eq!(mask_of(&second), vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn no_target_mask_means_no_loss_mask_for_prefix_lm_ids() {
        let tokens = vec![0, 1, 4, 2, 12, EOS];
        let mut l = LitDataLoader::<CpuRuntime>::new(tokens, 1, 5, CpuDevice::new());
        let batch = l.next_batch().expect("batch");
        assert!(batch.loss_mask.is_none());
    }
}
