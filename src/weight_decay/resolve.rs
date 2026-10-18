//! The rule a run uses: its config, or the rule of the checkpoint it resumes.

use super::rule::{WeightDecayRule, WeightDecaySpec};
use crate::config::{CheckpointMeta, TrainerConfig};
use crate::prelude::{Error, Result};

/// The rule a run at `start_step` inherits from the checkpoint meta at `meta_path`.
///
/// A run at step 0 inherits nothing: it has no optimizer history to match.
/// A resume without a meta inherits [`WeightDecaySpec::legacy`].
///
/// # Errors
/// Returns an error when the meta is unreadable.
pub fn inherited_weight_decay(
    meta_path: Option<&str>,
    start_step: usize,
) -> Result<Option<WeightDecaySpec>> {
    if start_step == 0 {
        return Ok(None);
    }
    let Some(path) = meta_path else {
        return Ok(Some(WeightDecaySpec::legacy()));
    };
    let meta = CheckpointMeta::load(path).map_err(|e| Error::TrainingError {
        reason: format!(
            "checkpoint meta '{path}' was not read: {e}. Restore the file to resume with its \
             weight decay rule."
        ),
    })?;
    Ok(Some(WeightDecaySpec::of_meta(&meta)))
}

/// The rule of a run with trainer `config` that resumes with rule `inherited`.
///
/// - A fresh run (`inherited` is `None`) uses the config rule, else `skip_1d`.
/// - A resume uses each config field that is set, else the inherited value.
/// - A resume whose result differs from `inherited` warns on stderr. Under
///   `headless` it is an error instead.
///
/// # Errors
/// Returns an error when [`WeightDecaySpec::check`] fails, or on a headless mismatch.
pub fn resolve_weight_decay(
    config: &TrainerConfig,
    inherited: Option<&WeightDecaySpec>,
    headless: bool,
) -> Result<WeightDecaySpec> {
    let Some(previous) = inherited else {
        let spec = WeightDecaySpec {
            rule: config
                .weight_decay_rule
                .unwrap_or(WeightDecayRule::FRESH_DEFAULT),
            no_decay_patterns: config.no_decay_patterns.clone().unwrap_or_default(),
        };
        spec.check()?;
        return Ok(spec);
    };
    let rule = config.weight_decay_rule.unwrap_or(previous.rule);
    // Inherited patterns carry over only to the rule that takes patterns.
    let no_decay_patterns = match (&config.no_decay_patterns, rule) {
        (Some(patterns), _) => patterns.clone(),
        (None, WeightDecayRule::Skip1d) => previous.no_decay_patterns.clone(),
        (None, _) => Vec::new(),
    };
    let spec = WeightDecaySpec {
        rule,
        no_decay_patterns,
    };
    spec.check()?;
    if spec != *previous {
        let message = format!(
            "trainer weight decay rule '{spec}' differs from the checkpoint's '{previous}'. The \
             resumed run decays other parameters than the run it continues, so it is not exact."
        );
        let keep = "Remove trainer.weight_decay_rule and trainer.no_decay_patterns to keep the \
                    checkpoint's rule";
        if headless {
            return Err(Error::TrainingError {
                reason: format!(
                    "{message} {keep}, or resume without --headless to accept the change."
                ),
            });
        }
        eprintln!("WARNING: {message} {keep}.");
    }
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LitConfig;

    fn spec(rule: WeightDecayRule, patterns: &[&str]) -> WeightDecaySpec {
        WeightDecaySpec {
            rule,
            no_decay_patterns: patterns.iter().map(|p| p.to_string()).collect(),
        }
    }

    fn config(rule: Option<WeightDecayRule>, patterns: Option<&[&str]>) -> TrainerConfig {
        let mut trainer = LitConfig::nano_baseline().trainer;
        trainer.weight_decay_rule = rule;
        trainer.no_decay_patterns = patterns.map(|p| p.iter().map(|s| s.to_string()).collect());
        trainer
    }

    #[test]
    fn a_fresh_run_defaults_to_skip_1d() {
        let fresh = resolve_weight_decay(&config(None, None), None, true).unwrap();
        assert_eq!(fresh, spec(WeightDecayRule::Skip1d, &[]));
        let legacy = config(Some(WeightDecayRule::DecayAll), None);
        let explicit = resolve_weight_decay(&legacy, None, true).unwrap();
        assert_eq!(explicit, WeightDecaySpec::legacy());
    }

    #[test]
    fn a_resume_inherits_unset_fields() {
        let unset = config(None, None);
        let previous = spec(WeightDecayRule::Skip1d, &["embed"]);
        let resolved = resolve_weight_decay(&unset, Some(&previous), true).unwrap();
        assert_eq!(resolved, previous);
        let legacy = WeightDecaySpec::legacy();
        let resolved = resolve_weight_decay(&unset, Some(&legacy), true).unwrap();
        assert_eq!(resolved, legacy);
    }

    #[test]
    fn an_explicit_mismatch_errors_headless_and_wins_interactive() {
        let previous = WeightDecaySpec::legacy();
        let changed = config(Some(WeightDecayRule::Skip1d), None);
        let err = resolve_weight_decay(&changed, Some(&previous), true).unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("skip_1d") && text.contains("decay_all"),
            "{text}"
        );
        assert!(text.contains("--headless"), "{text}");
        let interactive = resolve_weight_decay(&changed, Some(&previous), false).unwrap();
        assert_eq!(interactive, spec(WeightDecayRule::Skip1d, &[]));

        let matching = config(Some(WeightDecayRule::DecayAll), None);
        let resolved = resolve_weight_decay(&matching, Some(&previous), true).unwrap();
        assert_eq!(resolved, previous);
    }

    #[test]
    fn a_changed_pattern_list_is_a_mismatch() {
        let previous = spec(WeightDecayRule::Skip1d, &["embed"]);
        let changed = config(None, Some(&["lora_"]));
        assert!(resolve_weight_decay(&changed, Some(&previous), true).is_err());
        let skip_all = config(Some(WeightDecayRule::SkipAll), None);
        let resolved = resolve_weight_decay(&skip_all, Some(&previous), false).unwrap();
        assert_eq!(resolved, spec(WeightDecayRule::SkipAll, &[]));
    }

    #[test]
    fn unset_fields_stay_out_of_a_serialized_config() {
        let yaml = serde_yaml::to_string(&config(None, None)).unwrap();
        assert!(!yaml.contains("weight_decay_rule"), "{yaml}");
        assert!(!yaml.contains("no_decay_patterns"), "{yaml}");
        let set = config(Some(WeightDecayRule::SkipAll), Some(&["x"]));
        let yaml = serde_yaml::to_string(&set).unwrap();
        assert!(yaml.contains("weight_decay_rule: skip_all"), "{yaml}");
        let back: TrainerConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(back.weight_decay_rule, Some(WeightDecayRule::SkipAll));
        assert_eq!(back.no_decay_patterns, Some(vec!["x".to_string()]));
    }

    #[test]
    fn an_invalid_config_errors_on_a_fresh_run() {
        let patterns = config(Some(WeightDecayRule::DecayAll), Some(&["embed"]));
        assert!(resolve_weight_decay(&patterns, None, false).is_err());
    }

    #[test]
    fn step_zero_inherits_nothing_and_a_missing_meta_inherits_legacy() {
        assert_eq!(
            inherited_weight_decay(Some("unread.json"), 0).unwrap(),
            None
        );
        assert_eq!(
            inherited_weight_decay(None, 3).unwrap(),
            Some(WeightDecaySpec::legacy())
        );
        let err = inherited_weight_decay(Some("no/such/meta.json"), 3).unwrap_err();
        assert!(err.to_string().contains("no/such/meta.json"), "{err}");
    }
}
