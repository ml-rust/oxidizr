//! The one naming difference between an oxidizr checkpoint's on-disk tensor
//! keys and [`crate::model::LitGPT::named_parameters`]'s own names.
//!
//! HuggingFace, and boostr's loaders (`vb.pp("model")` in the llama, mamba,
//! and hybrid loaders), keep the decoder body under a `model.` prefix and the
//! LM head at the root: `model.embed_tokens.weight`, `model.layers.0...`,
//! `model.norm.weight`, `lm_head.weight`. `LitGPT::named_parameters` has no
//! such prefix. [`to_checkpoint_name`] and [`from_checkpoint_name`] are the
//! ONLY place this mapping is applied — every writer and reader goes through
//! them so the rule cannot drift between call sites.

/// HuggingFace's, and boostr's loaders', prefix on every decoder parameter.
/// `lm_head.weight` is the one tensor outside it.
pub const CHECKPOINT_DECODER_PREFIX: &str = "model.";

/// Map an oxidizr parameter name (as [`crate::model::LitGPT::named_parameters`]
/// produces it) to the name it is written under on disk.
///
/// `lm_head.weight` stays at the root; every other name gets the `model.`
/// prefix, LoRA adapter factors (`...lora_a`/`...lora_b`) included, since they
/// live under the same decoder-body names as the base weights they adapt.
pub fn to_checkpoint_name(param_name: &str) -> String {
    if param_name.starts_with("lm_head.") {
        param_name.to_string()
    } else {
        format!("{CHECKPOINT_DECODER_PREFIX}{param_name}")
    }
}

/// Map an on-disk checkpoint tensor name back to an oxidizr parameter name.
///
/// Strips a leading `model.` when present; a name without it — an older,
/// unprefixed checkpoint, or `lm_head.weight` — passes through unchanged.
pub fn from_checkpoint_name(name: &str) -> &str {
    name.strip_prefix(CHECKPOINT_DECODER_PREFIX).unwrap_or(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_body_round_trips_through_the_prefix() {
        let name = "layers.0.self_attn.q_proj.weight";
        let on_disk = to_checkpoint_name(name);
        assert_eq!(on_disk, "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(from_checkpoint_name(&on_disk), name);
    }

    #[test]
    fn lm_head_stays_at_the_root() {
        let name = "lm_head.weight";
        let on_disk = to_checkpoint_name(name);
        assert_eq!(on_disk, "lm_head.weight");
        assert_eq!(from_checkpoint_name(&on_disk), name);
    }

    #[test]
    fn a_lora_adapter_name_round_trips_like_any_other_decoder_parameter() {
        let name = "layers.3.self_attn.q_proj.lora_a";
        let on_disk = to_checkpoint_name(name);
        assert_eq!(on_disk, "model.layers.3.self_attn.q_proj.lora_a");
        assert_eq!(from_checkpoint_name(&on_disk), name);
    }

    #[test]
    fn an_unprefixed_legacy_name_passes_through_unchanged() {
        assert_eq!(
            from_checkpoint_name("embed_tokens.weight"),
            "embed_tokens.weight"
        );
    }
}
