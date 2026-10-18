//! Writes the checkpoint directory's `config.json`.
//!
//! The file is boostr's own [`UniversalConfig`](boostr::model::UniversalConfig),
//! derived from the trained [`OxidizrModelConfig`] through
//! [`OxidizrModelConfig::to_boostr`] — the same conversion `LitGPT::new` uses
//! to build the model being trained. blazr's `BlazrConfig` deserializes
//! `UniversalConfig` directly (`#[serde(flatten)]`), so this file loads with
//! no reader-side guessing of `rope_theta`, head counts, or SSM/MoE/hybrid
//! shape: every architecture field the checkpoint needs is present.

use crate::config::OxidizrModelConfig;
use anyhow::Context;
use std::path::{Path, PathBuf};

/// `config.json`'s path inside a checkpoint directory.
fn config_path(checkpoint_dir: &str) -> PathBuf {
    Path::new(checkpoint_dir).join("config.json")
}

/// True when `checkpoint_dir` already holds a `config.json`.
pub fn checkpoint_config_exists(checkpoint_dir: &str) -> bool {
    config_path(checkpoint_dir).exists()
}

/// Derives boostr's `UniversalConfig` from `model_config` and writes it as
/// `checkpoint_dir/config.json`.
pub fn write_checkpoint_config(
    model_config: &OxidizrModelConfig,
    checkpoint_dir: &str,
) -> anyhow::Result<()> {
    let universal = model_config
        .to_boostr()
        .context("failed to derive boostr UniversalConfig for checkpoint config.json")?;
    let json = serde_json::to_string_pretty(&universal)
        .context("failed to serialize checkpoint config.json")?;
    std::fs::write(config_path(checkpoint_dir), json)
        .context("failed to write checkpoint config.json")?;
    Ok(())
}

/// Sets an absent `moe.shared_expert` to `true` in a `config.json` from an
/// older oxidizr. Returns whether it changed `config`.
///
/// Older oxidizr wrote the flag as `null` and always trained a shared expert.
/// boostr reads `null` as no shared expert, so the weights and the config
/// disagreed. Current writes always carry the flag.
pub fn repair_legacy_shared_expert(config: &mut boostr::model::UniversalConfig) -> bool {
    match config.moe.as_mut() {
        Some(moe) if moe.shared_expert.is_none() => {
            moe.shared_expert = Some(true);
            true
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::model::UniversalConfig;

    /// Parses only a training YAML's `model:` section into
    /// `OxidizrModelConfig`, sidestepping `trainer:` (checkpoint paths, `${VAR}`
    /// placeholders, sidecar files) entirely — this test checks architecture
    /// derivation only.
    fn load_model_config(yaml_path: &str) -> OxidizrModelConfig {
        let content =
            std::fs::read_to_string(yaml_path).unwrap_or_else(|e| panic!("read {yaml_path}: {e}"));
        let full: serde_yaml::Value =
            serde_yaml::from_str(&content).unwrap_or_else(|e| panic!("parse {yaml_path}: {e}"));
        let model = full
            .get("model")
            .unwrap_or_else(|| panic!("{yaml_path} has no 'model:' section"));
        serde_yaml::from_value(model.clone())
            .unwrap_or_else(|e| panic!("parse model section of {yaml_path}: {e}"))
    }

    /// A checkpoint directory scoped to this test process and test name, under
    /// the system temp dir. Removed at the end of the test.
    fn scratch_checkpoint_dir(test_name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("oxidizr-checkpoint-config-test-{test_name}"))
    }

    /// Writes `model_config`'s checkpoint config.json, reads it back as
    /// `UniversalConfig`, and asserts every architecture field the training
    /// config set survived: model type, dimensions, attention/SSM/MoE/hybrid
    /// shape.
    fn assert_round_trips(model_config: &OxidizrModelConfig, test_name: &str) {
        let dir = scratch_checkpoint_dir(test_name);
        std::fs::create_dir_all(&dir).expect("create scratch checkpoint dir");

        write_checkpoint_config(model_config, dir.to_str().expect("utf8 path"))
            .expect("write_checkpoint_config must succeed for a valid model config");

        let content = std::fs::read_to_string(dir.join("config.json")).expect("read config.json");
        let loaded: UniversalConfig =
            serde_json::from_str(&content).expect("config.json must parse as UniversalConfig");

        let expected = model_config
            .to_boostr()
            .expect("to_boostr must succeed for a valid model config");

        assert_eq!(loaded.model_type, expected.model_type);
        assert_eq!(loaded.vocab_size, expected.vocab_size);
        assert_eq!(loaded.hidden_size, expected.hidden_size);
        assert_eq!(loaded.num_layers, expected.num_layers);
        assert_eq!(loaded.max_seq_len, expected.max_seq_len);
        assert_eq!(loaded.tie_word_embeddings, expected.tie_word_embeddings);

        match (&loaded.attention, &expected.attention) {
            (Some(got), Some(want)) => {
                assert_eq!(got.num_heads, want.num_heads);
                assert_eq!(got.num_kv_heads, want.num_kv_heads);
                assert_eq!(got.head_dim, want.head_dim);
                assert_eq!(got.rope_theta, want.rope_theta);
                assert_eq!(got.kv_latent_dim, want.kv_latent_dim);
                assert_eq!(got.q_latent_dim, want.q_latent_dim);
                assert_eq!(got.d_rope, want.d_rope);
            }
            (None, None) => {}
            (got, want) => panic!("attention presence mismatch: got {got:?}, want {want:?}"),
        }

        match (&loaded.ssm, &expected.ssm) {
            (Some(got), Some(want)) => {
                assert_eq!(got.variant, want.variant);
                assert_eq!(got.num_heads, want.num_heads);
                assert_eq!(got.head_dim, want.head_dim);
                assert_eq!(got.state_size, want.state_size);
                assert_eq!(got.chunk_size, want.chunk_size);
            }
            (None, None) => {}
            (got, want) => panic!("ssm presence mismatch: got {got:?}, want {want:?}"),
        }

        match (&loaded.moe, &expected.moe) {
            (Some(got), Some(want)) => {
                assert_eq!(got.num_experts, want.num_experts);
                assert_eq!(got.experts_per_tok, want.experts_per_tok);
            }
            (None, None) => {}
            (got, want) => panic!("moe presence mismatch: got {got:?}, want {want:?}"),
        }

        match (&loaded.hybrid_layers, &expected.hybrid_layers) {
            (Some(got), Some(want)) => {
                assert_eq!(got.ssm_layers, want.ssm_layers);
                assert_eq!(got.attention_layers, want.attention_layers);
            }
            (None, None) => {}
            (got, want) => panic!("hybrid_layers presence mismatch: got {got:?}, want {want:?}"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    /// `models/nano.yaml`: hybrid Mamba2 + MLA + fine-grained MoE. Exercises
    /// ssm, attention (MLA fields), moe, and hybrid_layers all at once.
    #[test]
    fn nano_config_round_trips_through_universal_config() {
        let model_config = load_model_config("models/nano.yaml");
        assert_round_trips(&model_config, "nano");
    }

    /// `models/g2p-char/g2p-char.yaml`: dense LLaMA-style attention, no
    /// Mamba/MLA/MoE. Exercises the attention-only, non-GQA path.
    #[test]
    fn g2p_char_config_round_trips_through_universal_config() {
        let model_config = load_model_config("models/g2p-char/g2p-char.yaml");
        assert_round_trips(&model_config, "g2p-char");
    }

    #[test]
    fn checkpoint_config_exists_reflects_the_written_file() {
        let dir = scratch_checkpoint_dir("exists-check");
        std::fs::create_dir_all(&dir).expect("create scratch checkpoint dir");
        assert!(!checkpoint_config_exists(dir.to_str().unwrap()));

        let model_config = load_model_config("models/g2p-char/g2p-char.yaml");
        write_checkpoint_config(&model_config, dir.to_str().unwrap()).expect("write must succeed");
        assert!(checkpoint_config_exists(dir.to_str().unwrap()));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// An MoE config written by an older oxidizr carries `shared_expert: null`.
    #[test]
    fn a_null_shared_expert_flag_is_repaired_to_true_and_an_explicit_one_is_kept() {
        let mut model_config = load_model_config("models/nano.yaml");
        let mut legacy = model_config.to_boostr().expect("nano converts");
        legacy.moe.as_mut().expect("nano is MoE").shared_expert = None;
        assert!(repair_legacy_shared_expert(&mut legacy));
        assert_eq!(
            legacy.moe.as_ref().and_then(|m| m.shared_expert),
            Some(true)
        );

        model_config.shared_expert_enabled = Some(false);
        let mut explicit = model_config.to_boostr().expect("nano converts");
        assert!(!repair_legacy_shared_expert(&mut explicit));
        assert_eq!(
            explicit.moe.as_ref().and_then(|m| m.shared_expert),
            Some(false)
        );
    }
}
