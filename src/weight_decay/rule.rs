//! The weight decay rule and the per-parameter test it applies.

use crate::config::CheckpointMeta;
use crate::prelude::{Error, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Which parameters AdamW weight decay applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightDecayRule {
    /// Decay every parameter. Runs from before parameter groups used this rule.
    #[serde(rename = "decay_all")]
    DecayAll,
    /// No decay on tensors of rank < 2, on `*bias` tensors, and on the SSM
    /// `a_log` and `D` tensors. Every other tensor decays: embeddings, LoRA
    /// factors, and projection weights.
    #[serde(rename = "skip_1d")]
    Skip1d,
    /// No decay on any parameter.
    #[serde(rename = "skip_all")]
    SkipAll,
}

impl WeightDecayRule {
    /// The rule of a fresh run whose config sets none.
    pub const FRESH_DEFAULT: Self = Self::Skip1d;
    /// The rule of a checkpoint meta that records none.
    pub const LEGACY: Self = Self::DecayAll;

    /// The YAML spelling of the rule.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DecayAll => "decay_all",
            Self::Skip1d => "skip_1d",
            Self::SkipAll => "skip_all",
        }
    }
}

impl fmt::Display for WeightDecayRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Last name segments of the SSM tensors `skip_1d` keeps out of decay at any rank.
///
/// Mamba1 `a_log` is `[d_inner, d_state]`, so the rank test alone decays it.
const SSM_NO_DECAY: [&str; 4] = ["a_log", "A_log", "d_param", "D"];

/// The resolved weight decay rule of one run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightDecaySpec {
    pub rule: WeightDecayRule,
    /// Name substrings kept out of decay on top of `skip_1d`.
    pub no_decay_patterns: Vec<String>,
}

impl WeightDecaySpec {
    /// The rule of a run that predates parameter groups: decay everything.
    pub fn legacy() -> Self {
        Self {
            rule: WeightDecayRule::LEGACY,
            no_decay_patterns: Vec::new(),
        }
    }

    /// The rule `meta` records. A meta without the field reads as [`Self::legacy`].
    pub fn of_meta(meta: &CheckpointMeta) -> Self {
        Self {
            rule: meta.weight_decay_rule.unwrap_or(WeightDecayRule::LEGACY),
            no_decay_patterns: meta.no_decay_patterns.clone(),
        }
    }

    /// Check the patterns against the rule.
    ///
    /// # Errors
    /// Returns an error for an empty pattern, or for patterns under a rule other than `skip_1d`.
    pub fn check(&self) -> Result<()> {
        if let Some(i) = self.no_decay_patterns.iter().position(String::is_empty) {
            return Err(Error::TrainingError {
                reason: format!(
                    "trainer.no_decay_patterns[{i}] is empty, and an empty pattern matches every \
                     parameter. Remove it, or set trainer.weight_decay_rule: skip_all."
                ),
            });
        }
        if !self.no_decay_patterns.is_empty() && self.rule != WeightDecayRule::Skip1d {
            return Err(Error::TrainingError {
                reason: format!(
                    "trainer.no_decay_patterns {:?} needs weight_decay_rule skip_1d, but the rule \
                     is {}. Set trainer.weight_decay_rule: skip_1d, or remove \
                     trainer.no_decay_patterns.",
                    self.no_decay_patterns, self.rule
                ),
            });
        }
        Ok(())
    }

    /// True when the parameter with checkpoint name `name` and `rank` takes no weight decay.
    pub fn excludes(&self, name: &str, rank: usize) -> bool {
        match self.rule {
            WeightDecayRule::DecayAll => false,
            WeightDecayRule::SkipAll => true,
            WeightDecayRule::Skip1d => {
                let last = name.rsplit('.').next().unwrap_or(name);
                rank < 2
                    || last.ends_with("bias")
                    || SSM_NO_DECAY.contains(&last)
                    || self
                        .no_decay_patterns
                        .iter()
                        .any(|p| name.contains(p.as_str()))
            }
        }
    }
}

impl fmt::Display for WeightDecaySpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.rule)?;
        if !self.no_decay_patterns.is_empty() {
            write!(f, " with no_decay_patterns {:?}", self.no_decay_patterns)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(rule: WeightDecayRule, patterns: &[&str]) -> WeightDecaySpec {
        WeightDecaySpec {
            rule,
            no_decay_patterns: patterns.iter().map(|p| p.to_string()).collect(),
        }
    }

    const NAMES: [(&str, usize); 10] = [
        ("model.embed_tokens.weight", 2),
        ("model.layers.0.input_layernorm.weight", 1),
        ("model.layers.0.self_attn.q_proj.weight", 2),
        ("model.layers.0.self_attn.q_proj.bias", 1),
        ("model.layers.0.self_attn.q_proj.lora_a", 2),
        ("model.layers.1.mamba.a_log", 2),
        ("model.layers.1.mamba.d_param", 1),
        ("model.layers.1.mamba.dt_bias", 1),
        ("model.layers.1.mamba.b_bias", 2),
        ("lm_head.weight", 2),
    ];

    fn excluded(spec: &WeightDecaySpec) -> Vec<&'static str> {
        NAMES
            .iter()
            .filter(|(name, rank)| spec.excludes(name, *rank))
            .map(|(name, _)| *name)
            .collect()
    }

    #[test]
    fn skip_1d_keeps_norms_biases_and_ssm_tensors_out_of_decay() {
        assert_eq!(
            excluded(&spec(WeightDecayRule::Skip1d, &[])),
            vec![
                "model.layers.0.input_layernorm.weight",
                "model.layers.0.self_attn.q_proj.bias",
                "model.layers.1.mamba.a_log",
                "model.layers.1.mamba.d_param",
                "model.layers.1.mamba.dt_bias",
                "model.layers.1.mamba.b_bias",
            ]
        );
    }

    #[test]
    fn decay_all_and_skip_all_ignore_rank_and_name() {
        assert!(excluded(&spec(WeightDecayRule::DecayAll, &[])).is_empty());
        assert_eq!(
            excluded(&spec(WeightDecayRule::SkipAll, &[])).len(),
            NAMES.len()
        );
    }

    #[test]
    fn a_pattern_matches_a_substring_of_the_checkpoint_name() {
        let with = excluded(&spec(WeightDecayRule::Skip1d, &["embed_tokens", "lora_"]));
        assert!(with.contains(&"model.embed_tokens.weight"));
        assert!(with.contains(&"model.layers.0.self_attn.q_proj.lora_a"));
        assert!(!with.contains(&"lm_head.weight"));
        assert!(!with.contains(&"model.layers.0.self_attn.q_proj.weight"));
    }

    #[test]
    fn patterns_need_skip_1d_and_must_not_be_empty() {
        let err = spec(WeightDecayRule::DecayAll, &["embed"])
            .check()
            .unwrap_err();
        assert!(err.to_string().contains("skip_1d"), "{err}");
        let err = spec(WeightDecayRule::Skip1d, &[""]).check().unwrap_err();
        assert!(err.to_string().contains("no_decay_patterns[0]"), "{err}");
        assert!(spec(WeightDecayRule::Skip1d, &["embed"]).check().is_ok());
    }

    #[test]
    fn a_meta_without_the_rule_reads_as_decay_all() {
        let old = r#"{"step":3,"timestamp":"t","model_config_hash":"h","data_file_path":"",
            "data_file_hash":"","checkpoint_file":"w.safetensors"}"#;
        let meta: CheckpointMeta = serde_json::from_str(old).unwrap();
        assert_eq!(WeightDecaySpec::of_meta(&meta), WeightDecaySpec::legacy());

        let mut meta = meta;
        meta.weight_decay_rule = Some(WeightDecayRule::Skip1d);
        meta.no_decay_patterns = vec!["embed".to_string()];
        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains(r#""weight_decay_rule":"skip_1d""#), "{json}");
        let back: CheckpointMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(
            WeightDecaySpec::of_meta(&back),
            spec(WeightDecayRule::Skip1d, &["embed"])
        );
    }

    #[test]
    fn rule_spellings_round_trip_through_yaml() {
        for rule in [
            WeightDecayRule::DecayAll,
            WeightDecayRule::Skip1d,
            WeightDecayRule::SkipAll,
        ] {
            let text = serde_yaml::to_string(&rule).unwrap();
            assert_eq!(text.trim(), rule.as_str());
            let back: WeightDecayRule = serde_yaml::from_str(&text).unwrap();
            assert_eq!(back, rule);
        }
    }
}
