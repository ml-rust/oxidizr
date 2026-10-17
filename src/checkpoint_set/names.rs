//! The two name maps the optimizer state file is keyed by.
//!
//! The names are the checkpoint names of the weights file, so one tensor has
//! one name in both files.

use crate::checkpoint_names::to_checkpoint_name;
use crate::prelude::{Error, Result};
use boostr::tensor::TensorId;
use std::collections::{HashMap, HashSet};

/// Checkpoint name by parameter id, and parameter id by checkpoint name.
#[derive(Debug, Clone, Default)]
pub struct NameMaps {
    names: HashMap<TensorId, String>,
    ids: HashMap<String, TensorId>,
}

impl NameMaps {
    /// Build the maps for the parameters the optimizer steps.
    ///
    /// `parameter_names` lists `(name, id)` in `LitGPT::named_parameters`
    /// order. `parameter_ids` lists the ids the optimizer steps.
    /// - A name whose id is not in `parameter_ids` is frozen and has no state. It is skipped.
    /// - A second name for one id is a tied alias. The first name wins, as in the weights file.
    ///
    /// # Errors
    /// Returns an error when two ids map to one checkpoint name.
    pub fn new(parameter_names: &[(String, TensorId)], parameter_ids: &[TensorId]) -> Result<Self> {
        let trainable: HashSet<TensorId> = parameter_ids.iter().copied().collect();
        let mut maps = Self::default();
        for (name, id) in parameter_names {
            if !trainable.contains(id) || maps.names.contains_key(id) {
                continue;
            }
            let checkpoint_name = to_checkpoint_name(name);
            if let Some(previous) = maps.ids.insert(checkpoint_name.clone(), *id) {
                return Err(Error::TrainingError {
                    reason: format!(
                        "parameters {previous} and {id} both map to checkpoint name \
                         '{checkpoint_name}'. Rename one parameter so each checkpoint name is unique."
                    ),
                });
            }
            maps.names.insert(*id, checkpoint_name);
        }
        Ok(maps)
    }

    /// Checkpoint name by parameter id, as `Optimizer::state_dict` takes it.
    pub fn names(&self) -> &HashMap<TensorId, String> {
        &self.names
    }

    /// Parameter id by checkpoint name, as `Optimizer::load_state_dict` takes it.
    pub fn ids(&self) -> &HashMap<String, TensorId> {
        &self.ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_are_inverse_and_use_checkpoint_names() {
        let (a, b) = (TensorId::new(), TensorId::new());
        let names = vec![
            ("embed_tokens.weight".to_string(), a),
            ("lm_head.weight".to_string(), b),
        ];
        let maps = NameMaps::new(&names, &[a, b]).unwrap();
        assert_eq!(maps.names()[&a], "model.embed_tokens.weight");
        assert_eq!(maps.names()[&b], "lm_head.weight");
        assert_eq!(maps.ids()["model.embed_tokens.weight"], a);
        assert_eq!(maps.ids()["lm_head.weight"], b);
    }

    #[test]
    fn tied_alias_keeps_the_first_name() {
        let tied = TensorId::new();
        let names = vec![
            ("embed_tokens.weight".to_string(), tied),
            ("lm_head.weight".to_string(), tied),
        ];
        let maps = NameMaps::new(&names, &[tied]).unwrap();
        assert_eq!(maps.names().len(), 1);
        assert_eq!(maps.ids().len(), 1);
        assert_eq!(maps.names()[&tied], "model.embed_tokens.weight");
        assert!(!maps.ids().contains_key("lm_head.weight"));
    }

    #[test]
    fn frozen_parameters_are_skipped() {
        let (base, lora_a) = (TensorId::new(), TensorId::new());
        let names = vec![
            ("layers.0.self_attn.q_proj.weight".to_string(), base),
            ("layers.0.self_attn.q_proj.lora_a".to_string(), lora_a),
        ];
        let maps = NameMaps::new(&names, &[lora_a]).unwrap();
        assert_eq!(maps.names().len(), 1);
        assert_eq!(
            maps.names()[&lora_a],
            "model.layers.0.self_attn.q_proj.lora_a"
        );
        assert!(!maps.names().contains_key(&base));
    }

    #[test]
    fn two_ids_with_one_name_are_an_error() {
        let (a, b) = (TensorId::new(), TensorId::new());
        let names = vec![
            ("norm.weight".to_string(), a),
            ("norm.weight".to_string(), b),
        ];
        let err = NameMaps::new(&names, &[a, b]).unwrap_err();
        assert!(err.to_string().contains("model.norm.weight"), "{err}");
    }
}
