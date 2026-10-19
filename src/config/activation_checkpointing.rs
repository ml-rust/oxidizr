//! `trainer.activation_checkpointing`: which layers recompute their activations.
//!
//! A covered layer keeps only its input during the forward pass. The backward
//! pass runs the layer again to rebuild its graph. CPU tests check bit-identical
//! values and gradients. CUDA bitwise equality is unconfirmed. The key never
//! enters the model config hash or a checkpoint meta, and a resume can change it.
//!
//! YAML:
//!
//! ```yaml
//! activation_checkpointing: none          # the default
//! activation_checkpointing: all
//! activation_checkpointing: {layers: [0, 3, 5]}
//! ```

use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashSet;
use std::fmt;

/// The config key, as errors name it.
const KEY: &str = "trainer.activation_checkpointing";

/// Which layers recompute their activations in the backward pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ActivationCheckpointing {
    /// Every layer keeps its activations. The default.
    #[default]
    None,
    /// Every layer recomputes.
    All,
    /// The listed layer indices recompute.
    Layers(Vec<usize>),
}

impl ActivationCheckpointing {
    /// `true` for [`ActivationCheckpointing::None`]. The serde skip predicate.
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// `true` when layer `index` recomputes.
    pub fn covers(&self, index: usize) -> bool {
        match self {
            Self::None => false,
            Self::All => true,
            Self::Layers(layers) => layers.contains(&index),
        }
    }

    /// Check a `layers` list against a model of `num_layers` layers.
    ///
    /// # Errors
    ///
    /// - The list is empty.
    /// - An index is not below `num_layers`.
    /// - An index appears twice.
    pub fn check(&self, num_layers: usize) -> anyhow::Result<()> {
        let Self::Layers(layers) = self else {
            return Ok(());
        };
        if layers.is_empty() {
            anyhow::bail!(
                "{KEY}.layers is empty. List at least one index below model.num_layers \
                 ({num_layers}), or set {KEY} to none."
            );
        }
        let mut seen = HashSet::new();
        for &index in layers {
            if index >= num_layers {
                anyhow::bail!(
                    "{KEY}.layers contains {index}, which is not below model.num_layers \
                     ({num_layers}). Remove it or use an index from 0 to {}.",
                    num_layers.saturating_sub(1)
                );
            }
            if !seen.insert(index) {
                anyhow::bail!(
                    "{KEY}.layers lists {index} twice (model.num_layers {num_layers}). \
                     Remove the duplicate."
                );
            }
        }
        Ok(())
    }
}

impl Serialize for ActivationCheckpointing {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::None => serializer.serialize_str("none"),
            Self::All => serializer.serialize_str("all"),
            Self::Layers(layers) => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("layers", layers)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for ActivationCheckpointing {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(PolicyVisitor)
    }
}

/// Reads `none`, `all` or `{layers: [..]}`.
struct PolicyVisitor;

impl<'de> Visitor<'de> for PolicyVisitor {
    type Value = ActivationCheckpointing;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{KEY} as none, all, or {{layers: [0, 3, 5]}}")
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
        match value {
            "none" => Ok(ActivationCheckpointing::None),
            "all" => Ok(ActivationCheckpointing::All),
            other => Err(E::custom(format!(
                "{KEY} is `{other}`. Use none, all, or {{layers: [0, 3, 5]}}."
            ))),
        }
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut layers: Option<Vec<usize>> = None;
        while let Some(key) = map.next_key::<String>()? {
            if key != "layers" {
                return Err(de::Error::custom(format!(
                    "{KEY} has the key `{key}`. The only map key is `layers`."
                )));
            }
            if layers.is_some() {
                return Err(de::Error::custom(format!("{KEY} sets `layers` twice.")));
            }
            layers = Some(map.next_value()?);
        }
        layers
            .map(ActivationCheckpointing::Layers)
            .ok_or_else(|| de::Error::custom(format!("{KEY} is a map without `layers`.")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LitConfig;

    fn parse(yaml: &str) -> Result<ActivationCheckpointing, serde_yaml::Error> {
        serde_yaml::from_str(yaml)
    }

    #[test]
    fn the_three_yaml_forms_parse() {
        assert_eq!(parse("none").unwrap(), ActivationCheckpointing::None);
        assert_eq!(parse("all").unwrap(), ActivationCheckpointing::All);
        assert_eq!(
            parse("{layers: [0, 3, 5]}").unwrap(),
            ActivationCheckpointing::Layers(vec![0, 3, 5])
        );
        assert_eq!(
            parse("layers:\n  - 1\n  - 2\n").unwrap(),
            ActivationCheckpointing::Layers(vec![1, 2])
        );
    }

    #[test]
    fn an_unknown_form_names_the_key_and_the_accepted_forms() {
        let err = parse("some").unwrap_err().to_string();
        assert!(err.contains(KEY) && err.contains("`some`"), "{err}");
        let err = parse("{layer: [1]}").unwrap_err().to_string();
        assert!(err.contains("`layer`"), "{err}");
        let err = parse("{}").unwrap_err().to_string();
        assert!(err.contains("without `layers`"), "{err}");
    }

    #[test]
    fn the_trainer_key_defaults_to_none_when_absent() {
        let trainer = LitConfig::nano_baseline().trainer;
        let yaml = serde_yaml::to_string(&trainer).unwrap();
        let parsed: crate::config::TrainerConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(
            parsed.activation_checkpointing,
            ActivationCheckpointing::None
        );
    }

    /// The default stays out of serialized configs, so existing configs
    /// serialize as before. A set value round-trips.
    #[test]
    fn none_is_skipped_on_serialize_and_set_values_round_trip() {
        let mut trainer = LitConfig::nano_baseline().trainer;
        let json = serde_json::to_string(&trainer).unwrap();
        assert!(!json.contains("activation_checkpointing"), "{json}");
        let yaml = serde_yaml::to_string(&trainer).unwrap();
        assert!(!yaml.contains("activation_checkpointing"), "{yaml}");

        for policy in [
            ActivationCheckpointing::All,
            ActivationCheckpointing::Layers(vec![2, 0]),
        ] {
            trainer.activation_checkpointing = policy.clone();
            let yaml = serde_yaml::to_string(&trainer).unwrap();
            let parsed: crate::config::TrainerConfig = serde_yaml::from_str(&yaml).unwrap();
            assert_eq!(parsed.activation_checkpointing, policy);
            let json = serde_json::to_string(&trainer).unwrap();
            let parsed: crate::config::TrainerConfig = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.activation_checkpointing, policy);
        }
    }

    #[test]
    fn covers_follows_the_policy() {
        assert!(!ActivationCheckpointing::None.covers(0));
        assert!(ActivationCheckpointing::All.covers(7));
        let layers = ActivationCheckpointing::Layers(vec![1, 3]);
        assert!(layers.covers(1) && layers.covers(3));
        assert!(!layers.covers(0) && !layers.covers(2));
    }

    fn validate(policy: ActivationCheckpointing) -> anyhow::Result<()> {
        let mut cfg = LitConfig::nano_baseline();
        cfg.trainer.activation_checkpointing = policy;
        cfg.validate()
    }

    #[test]
    fn validation_accepts_in_range_lists() {
        let num_layers = LitConfig::nano_baseline().model.num_layers;
        validate(ActivationCheckpointing::None).unwrap();
        validate(ActivationCheckpointing::All).unwrap();
        validate(ActivationCheckpointing::Layers(vec![0, num_layers - 1])).unwrap();
    }

    #[test]
    fn validation_rejects_an_empty_list() {
        let err = validate(ActivationCheckpointing::Layers(Vec::new()))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("trainer.activation_checkpointing.layers is empty"),
            "{err}"
        );
    }

    #[test]
    fn validation_names_an_out_of_range_index_and_num_layers() {
        let num_layers = LitConfig::nano_baseline().model.num_layers;
        let err = validate(ActivationCheckpointing::Layers(vec![0, num_layers]))
            .unwrap_err()
            .to_string();
        let want = format!(
            "trainer.activation_checkpointing.layers contains {num_layers}, which is not \
             below model.num_layers ({num_layers})"
        );
        assert!(err.contains(&want), "{err}");
    }

    #[test]
    fn validation_names_a_duplicate_index() {
        let num_layers = LitConfig::nano_baseline().model.num_layers;
        let err = validate(ActivationCheckpointing::Layers(vec![1, 2, 1]))
            .unwrap_err()
            .to_string();
        let want = format!(
            "trainer.activation_checkpointing.layers lists 1 twice (model.num_layers {num_layers})"
        );
        assert!(err.contains(&want), "{err}");
    }
}
