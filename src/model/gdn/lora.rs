//! LoRA on a [`GdnLayer`]'s mixer: the base ids an adapter freezes, and the
//! merge that folds each adapter into its base.
//!
//! The FFN's adapters stay with the caller, as on the attention layers.

use super::layer::{mixer_prefix, GdnLayer};
use super::mixer;
use crate::config::LoraConfig;
use crate::model::{merge_projection, FrozenParams};
use crate::prelude::{DType, Result};
use boostr::autograd::TensorId;

impl<R> GdnLayer<R>
where
    R: boostr::Runtime<DType = DType>,
{
    /// The base-weight id of each adapted mixer projection.
    ///
    /// # Errors
    ///
    /// An adapted projection without its base weight.
    pub(in crate::model) fn adapted_mixer_base_ids(&self) -> Result<Vec<TensorId>> {
        mixer::adapted_base_ids(&self.block)
    }

    /// Fold each mixer adapter into its base weight. Returns how many
    /// projections merged.
    ///
    /// A merged projection holds `W + scaling * (B @ A)` under a fresh id,
    /// frozen, as `merge_projection` gives every other projection. Every
    /// other tensor keeps its id and flag.
    ///
    /// # Errors
    ///
    /// An adapted projection while `lora` is `None`, and the merge and
    /// rebuild errors.
    pub(in crate::model) fn merge_mixer_adapters<C>(
        &mut self,
        client: &C,
        lora: Option<&LoraConfig>,
        layer_idx: usize,
    ) -> Result<usize>
    where
        C: boostr::RuntimeClient<R>
            + boostr::TensorOps<R>
            + boostr::BinaryOps<R>
            + boostr::ScalarOps<R>,
        R::Client: boostr::TensorOps<R>
            + boostr::BinaryOps<R>
            + boostr::ScalarOps<R>
            + boostr::DequantOps<R>,
    {
        if self.adapted_mixer_base_ids()?.is_empty() {
            return Ok(0);
        }
        let prefix = mixer_prefix(layer_idx);
        let alpha = lora.map(|lora| lora.alpha);
        let weights = mixer::read(&self.block, alpha, &prefix, mixer::current)?;
        let frozen = FrozenParams::from_ids(
            self.block
                .parameters()
                .into_iter()
                .filter(|(_, var)| !var.requires_grad())
                .map(|(id, _)| id),
        );
        let config = self.block.config().clone();
        let mut merged = 0usize;
        let block = mixer::build_with(client, &config, weights, &prefix, &frozen, |mut proj| {
            if merge_projection(&mut proj, client)? {
                merged += 1;
            }
            Ok(proj)
        })?;
        self.block = block;
        Ok(merged)
    }
}
