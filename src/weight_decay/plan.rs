//! The no-decay set of one run, and the AdamW parameter group built from it.

use super::rule::WeightDecaySpec;
use crate::prelude::{Error, Result, Tensor};
use boostr::optimizer::ParamGroup;
use boostr::runtime::Runtime;
use boostr::tensor::TensorId;
use std::collections::{HashMap, HashSet};

/// One trainable tensor as the decay rule sees it.
#[derive(Debug, Clone, Copy)]
pub struct TensorInfo<'a> {
    pub id: TensorId,
    /// Checkpoint name, as the weights file writes it.
    pub name: &'a str,
    pub rank: usize,
    pub elements: usize,
}

/// The no-decay set of one run, with tensor and element counts per side.
#[derive(Debug, Clone)]
pub struct DecayPlan {
    spec: WeightDecaySpec,
    no_decay: HashSet<TensorId>,
    decay_tensors: usize,
    decay_elements: usize,
    no_decay_elements: usize,
}

fn unknown(id: &TensorId, what: &str, action: &str) -> Error {
    Error::TrainingError {
        reason: format!(
            "trainable parameter {id:?} has no {what}, so its weight decay group is unknown. \
             {action}"
        ),
    }
}

impl DecayPlan {
    /// Apply `spec` to `tensors`. A repeated id counts once, under its first entry.
    pub fn new(spec: WeightDecaySpec, tensors: &[TensorInfo<'_>]) -> Self {
        let mut seen = HashSet::new();
        let mut plan = Self {
            spec,
            no_decay: HashSet::new(),
            decay_tensors: 0,
            decay_elements: 0,
            no_decay_elements: 0,
        };
        for tensor in tensors.iter().filter(|t| seen.insert(t.id)) {
            if plan.spec.excludes(tensor.name, tensor.rank) {
                plan.no_decay.insert(tensor.id);
                plan.no_decay_elements += tensor.elements;
            } else {
                plan.decay_tensors += 1;
                plan.decay_elements += tensor.elements;
            }
        }
        plan
    }

    /// Apply `spec` to the trainable `ids`, with ranks from `params` and names from `names`.
    ///
    /// # Errors
    /// Returns an error when an id has no tensor in `params` or no name in `names`.
    pub fn for_params<R: Runtime>(
        spec: WeightDecaySpec,
        ids: &[TensorId],
        params: &HashMap<TensorId, Tensor<R>>,
        names: &HashMap<TensorId, String>,
    ) -> Result<Self> {
        let tensors = ids
            .iter()
            .map(|id| {
                let tensor = params.get(id).ok_or_else(|| {
                    unknown(
                        id,
                        "tensor in the parameter map",
                        "Build the map from the model's trainable parameters.",
                    )
                })?;
                let name = names.get(id).ok_or_else(|| {
                    unknown(
                        id,
                        "checkpoint name",
                        "Pass the model's named trainable parameters.",
                    )
                })?;
                Ok(TensorInfo {
                    id: *id,
                    name,
                    rank: tensor.shape().len(),
                    elements: tensor.elem_count(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::new(spec, &tensors))
    }

    pub fn spec(&self) -> &WeightDecaySpec {
        &self.spec
    }

    /// Ids that take no weight decay.
    pub fn no_decay_ids(&self) -> &HashSet<TensorId> {
        &self.no_decay
    }

    /// The AdamW groups of the plan: one zero-decay group, or none.
    ///
    /// An empty no-decay set gives no group, so every parameter takes the
    /// base path, as before groups existed.
    pub fn param_groups(&self) -> Vec<ParamGroup> {
        if self.no_decay.is_empty() {
            return Vec::new();
        }
        vec![ParamGroup {
            ids: self.no_decay.clone(),
            lr_scale: 1.0,
            weight_decay: Some(0.0),
        }]
    }

    /// One log line: the rule, and the tensor and element counts of each side.
    pub fn summary(&self, weight_decay: f64) -> String {
        format!(
            "Weight decay {weight_decay} with rule {}: {} tensors ({} params) decay, {} tensors \
             ({} params) do not",
            self.spec,
            self.decay_tensors,
            self.decay_elements,
            self.no_decay.len(),
            self.no_decay_elements
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::rule::WeightDecayRule;
    use super::*;

    fn info(id: TensorId, name: &str, rank: usize, elements: usize) -> TensorInfo<'_> {
        TensorInfo {
            id,
            name,
            rank,
            elements,
        }
    }

    #[test]
    fn a_tied_id_counts_once_and_the_group_has_zero_decay() {
        let (tied, norm) = (TensorId::new(), TensorId::new());
        let tensors = [
            info(tied, "model.embed_tokens.weight", 2, 6),
            info(tied, "lm_head.weight", 2, 6),
            info(norm, "model.norm.weight", 1, 3),
        ];
        let skip_1d = WeightDecaySpec {
            rule: WeightDecayRule::Skip1d,
            no_decay_patterns: Vec::new(),
        };
        let plan = DecayPlan::new(skip_1d, &tensors);
        assert_eq!(plan.no_decay_ids(), &HashSet::from([norm]));
        let groups = plan.param_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].ids, HashSet::from([norm]));
        assert_eq!(groups[0].weight_decay, Some(0.0));
        assert_eq!(groups[0].lr_scale, 1.0);
        let summary = plan.summary(0.1);
        assert!(summary.contains("skip_1d"), "{summary}");
        assert!(summary.contains("1 tensors (6 params) decay"), "{summary}");
        assert!(summary.contains("1 tensors (3 params) do not"), "{summary}");
    }

    #[test]
    fn decay_all_builds_no_group() {
        let tensors = [info(TensorId::new(), "model.norm.weight", 1, 3)];
        let plan = DecayPlan::new(WeightDecaySpec::legacy(), &tensors);
        assert!(plan.no_decay_ids().is_empty());
        assert!(plan.param_groups().is_empty());
    }

    #[test]
    fn a_missing_name_is_an_error() {
        let device = boostr::CpuDevice::new();
        let tensor = Tensor::<boostr::CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap();
        let id = tensor.id();
        let params = HashMap::from([(id, tensor)]);
        let err = DecayPlan::for_params(WeightDecaySpec::legacy(), &[id], &params, &HashMap::new())
            .unwrap_err();
        assert!(err.to_string().contains("checkpoint name"), "{err}");
    }
}
