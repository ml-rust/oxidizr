//! Write one checkpoint set in commit order: optimizer state, weights, meta.

use super::files::CheckpointLabel;
use super::names::NameMaps;
use super::optimizer_io::write_optimizer_file;
use crate::config::{CheckpointMeta, OxidizrModelConfig};
use crate::data::DataPosition;
use crate::eval::EvalCheckpoint;
use crate::metrics::SpikeGuardState;
use crate::model::{save_checkpoint_from_params, LitGPT};
use crate::prelude::{DType, FusedAdamW, Result, Tensor};
use crate::weight_decay::WeightDecaySpec;
use boostr::ops::TensorOps;
use boostr::optimizer::GradScalerState;
use boostr::runtime::{Runtime, RuntimeClient};
use boostr::tensor::TensorId;
use boostr::TypeConversionOps;
use std::collections::HashMap;
use std::path::Path;

/// Everything one checkpoint set is written from.
pub struct CheckpointSave<'a, R: Runtime<DType = DType>> {
    /// Checkpoint directory. It is created when absent.
    pub dir: &'a Path,
    /// `OxidizrModelConfig::checkpoint_prefix` of the model.
    pub prefix: &'a str,
    pub label: CheckpointLabel,
    /// Completed optimizer steps, as the meta records them.
    pub step: usize,
    /// Loader position after `step` steps.
    pub position: DataPosition,
    pub model_config: &'a OxidizrModelConfig,
    /// Training data file. `None` for a run on generated data.
    pub data_file_path: Option<&'a str>,
    /// Source of the frozen parameters.
    pub model: &'a LitGPT<R>,
    /// The optimizer's parameter map.
    pub params: &'a HashMap<TensorId, Tensor<R>>,
    pub optimizer: &'a FusedAdamW<R>,
    pub name_maps: &'a NameMaps,
    /// Weight decay rule of the run, recorded in the meta for a resume to inherit.
    pub weight_decay: &'a WeightDecaySpec,
    /// Loss scaler state after `step` steps. `None` for a run without loss scaling.
    pub loss_scale: Option<GradScalerState>,
    /// Held-out rows and eval stop counters after `step` steps. Default without an eval.
    pub eval: EvalCheckpoint,
    /// Data-parallel ranks of the run. 1 without data parallelism.
    pub world_size: usize,
    /// Rows of one global batch over every rank, the batch `position` counts.
    pub global_batch_rows: usize,
    /// Spike guard EMA and counters after `step` steps. `None` without a guard.
    pub spike_guard: Option<SpikeGuardState>,
}

/// Write the optimizer state, then the weights, then the meta.
///
/// Each file is written atomically. The meta is the commit marker. A crash
/// before it leaves a set that `--resume auto` skips. A meta write error is a
/// warning: the weights stay usable through an explicit `--resume <path>`.
/// A set written over an older one, such as `best`, loses the old meta first,
/// so no meta ever names a half-replaced set.
///
/// # Errors
/// Returns an error when the directory, the old meta removal, the optimizer
/// state, or the weights write fails.
pub fn write_checkpoint_set<R>(save: CheckpointSave<'_, R>) -> Result<()>
where
    R: Runtime<DType = DType>,
    R::Client: RuntimeClient<R> + TensorOps<R> + TypeConversionOps<R>,
{
    std::fs::create_dir_all(save.dir)?;
    let meta_path = save.dir.join(save.label.meta_file_name());
    if let Err(e) = std::fs::remove_file(&meta_path) {
        if e.kind() != std::io::ErrorKind::NotFound {
            return Err(e.into());
        }
    }
    let optimizer_name = save.label.optimizer_file_name(save.prefix);
    write_optimizer_file(
        &save.dir.join(&optimizer_name),
        save.optimizer,
        save.name_maps,
    )?;

    let weights_path = save.dir.join(save.label.weights_file_name(save.prefix));
    let weights = weights_path.to_string_lossy().into_owned();
    save_checkpoint_from_params(&weights, save.model, save.params)?;

    let written = CheckpointMeta::new(
        save.step,
        save.model_config,
        save.data_file_path,
        &weights,
        Some(save.position),
    )
    .and_then(|mut meta| {
        meta.optimizer_file = Some(optimizer_name);
        meta.weight_decay_rule = Some(save.weight_decay.rule);
        meta.no_decay_patterns = save.weight_decay.no_decay_patterns.clone();
        meta.loss_scale = save.loss_scale;
        meta.holdout_rows = save.eval.holdout_rows;
        meta.best_eval_loss = save.eval.best_eval_loss;
        meta.evals_since_best = save.eval.evals_since_best;
        meta.world_size = Some(save.world_size);
        meta.global_batch_rows = Some(save.global_batch_rows);
        meta.spike_guard = save.spike_guard;
        meta.save_to(&meta_path)
    });
    if let Err(e) = written {
        eprintln!(
            "Warning: checkpoint meta '{}' was not written: {e}. `--resume auto` skips this \
             checkpoint. Pass '{weights}' to `--resume` to use it.",
            meta_path.display()
        );
    }
    Ok(())
}
