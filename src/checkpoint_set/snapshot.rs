//! Copy one checkpoint set to host memory on the training thread.
//!
//! A [`SetSnapshot`] holds CPU tensors, paths and a meta only. Any thread can
//! commit it with [`commit_set`](super::commit_set), and no runtime handle
//! crosses that thread boundary.

use super::names::NameMaps;
use super::optimizer_io::stage_optimizer_state;
use crate::config::{CheckpointMeta, OxidizrModelConfig};
use crate::data::{DataPosition, DataSourceRecord};
use crate::eval::EvalCheckpoint;
use crate::metrics::SpikeGuardState;
use crate::model::{stage_checkpoint_tensors, LitGPT, StageScope};
use crate::prelude::{DType, Error, FusedAdamW, Result, Tensor};
use crate::weight_decay::WeightDecaySpec;
use boostr::ops::TensorOps;
use boostr::optimizer::GradScalerState;
use boostr::runtime::{Runtime, RuntimeClient};
use boostr::tensor::TensorId;
use boostr::{CpuRuntime, TypeConversionOps};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

/// Host tensors keyed by checkpoint name.
pub type HostTensors = HashMap<String, Tensor<CpuRuntime>>;

/// One checkpoint set in host memory, ready to commit under any label.
///
/// The meta holds every field except `checkpoint_file`, `optimizer_file` and
/// `timestamp`. The commit of a label fills those three.
pub struct SetSnapshot {
    dir: PathBuf,
    prefix: String,
    step: usize,
    weights: HostTensors,
    /// `None` once the last label wrote its optimizer file.
    optimizer: Mutex<Option<HostTensors>>,
    meta: CheckpointMeta,
}

/// The writer thread receives snapshots by `Arc`, so a snapshot must be `Send + Sync`.
fn assert_send<T: Send + Sync>() {}
const _: fn() = assert_send::<SetSnapshot>;

impl SetSnapshot {
    /// A snapshot of `step` from host tensors, for the directory `dir` and checkpoint `prefix`.
    pub fn from_parts(
        dir: PathBuf,
        prefix: String,
        step: usize,
        weights: HostTensors,
        optimizer: HostTensors,
        meta: CheckpointMeta,
    ) -> Self {
        Self {
            dir,
            prefix,
            step,
            weights,
            optimizer: Mutex::new(Some(optimizer)),
            meta,
        }
    }

    /// Checkpoint directory of the set.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// `OxidizrModelConfig::checkpoint_prefix` of the model.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Completed optimizer steps of the set.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Every weights tensor, frozen ones included.
    pub fn weights(&self) -> &HostTensors {
        &self.weights
    }

    /// The meta without its label fields.
    pub fn meta(&self) -> &CheckpointMeta {
        &self.meta
    }

    /// Run `write` on the optimizer state. `release` drops the state after it.
    ///
    /// # Errors
    /// Returns an error when an earlier commit released the state or
    /// panicked while it held the state, or when `write` fails.
    pub fn with_optimizer(
        &self,
        release: bool,
        write: impl FnOnce(&HostTensors) -> Result<()>,
    ) -> Result<()> {
        let mut slot = self.optimizer_slot()?;
        let state = slot.as_ref().ok_or_else(|| Error::TrainingError {
            reason: format!(
                "optimizer state of the step {} checkpoint was released before this commit. \
                 Commit each label of a snapshot before the last one releases it.",
                self.step
            ),
        })?;
        write(state)?;
        if release {
            *slot = None;
        }
        Ok(())
    }

    fn optimizer_slot(&self) -> Result<MutexGuard<'_, Option<HostTensors>>> {
        self.optimizer.lock().map_err(|_| Error::TrainingError {
            reason: format!(
                "optimizer state of the step {} checkpoint is unusable: an earlier commit \
                 panicked while it wrote the file. Save the checkpoint again.",
                self.step
            ),
        })
    }
}

/// Host copies of the frozen parameters, staged once and shared by every snapshot.
///
/// A LoRA base weight never changes during a run, so one copy serves every save.
/// The cache restages when the frozen parameter ids differ from its copy.
#[derive(Default)]
pub struct FrozenCache {
    slot: Mutex<Option<Arc<FrozenHost>>>,
}

struct FrozenHost {
    ids: Vec<TensorId>,
    tensors: HostTensors,
}

impl FrozenCache {
    /// The host copy of the frozen parameters of `model`, staged on first use.
    fn host<R>(
        &self,
        source: &str,
        model: &LitGPT<R>,
        params: &HashMap<TensorId, Tensor<R>>,
    ) -> Result<Arc<FrozenHost>>
    where
        R: Runtime<DType = DType>,
        R::Client: RuntimeClient<R> + TensorOps<R> + TypeConversionOps<R>,
    {
        let ids: Vec<TensorId> = model
            .named_parameters()?
            .into_iter()
            .map(|(_, id, _)| id)
            .filter(|id| !params.contains_key(id))
            .collect();
        let mut slot = self.slot.lock().map_err(|_| Error::TrainingError {
            reason: "frozen parameter cache is unusable: an earlier save panicked while it \
                     staged the frozen parameters. Restart the run from its last checkpoint."
                .to_string(),
        })?;
        if let Some(host) = slot.as_ref().filter(|host| host.ids == ids) {
            return Ok(Arc::clone(host));
        }
        let tensors = stage_checkpoint_tensors(source, model, params, StageScope::Frozen)?;
        let host = Arc::new(FrozenHost { ids, tensors });
        *slot = Some(Arc::clone(&host));
        Ok(host)
    }
}

/// Everything one snapshot is copied from.
pub struct SnapshotInput<'a, R: Runtime<DType = DType>> {
    /// Checkpoint directory. The commit creates it when absent.
    pub dir: &'a Path,
    /// `OxidizrModelConfig::checkpoint_prefix` of the model.
    pub prefix: &'a str,
    /// Completed optimizer steps, as the meta records them.
    pub step: usize,
    /// Loader position after `step` steps.
    pub position: DataPosition,
    /// Model config, recorded in the meta.
    pub model_config: &'a OxidizrModelConfig,
    /// Training data sources in stream order. Empty for a run on generated data.
    pub data_sources: &'a [DataSourceRecord],
    /// Source of the frozen parameters.
    pub model: &'a LitGPT<R>,
    /// The optimizer's parameter map.
    pub params: &'a HashMap<TensorId, Tensor<R>>,
    /// Source of the optimizer state.
    pub optimizer: &'a FusedAdamW<R>,
    /// Checkpoint names of the optimizer's parameters.
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
    /// Host copies of the frozen parameters.
    pub frozen: &'a FrozenCache,
}

/// Copy the weights, the optimizer state and the meta of `input` to host memory.
///
/// Call it on the training thread between steps. The snapshot shares no
/// buffer with the device state, so the next step can run during its commit.
///
/// # Errors
/// Returns an error when a copy to host memory fails.
pub fn snapshot_set<R>(input: SnapshotInput<'_, R>) -> Result<SetSnapshot>
where
    R: Runtime<DType = DType>,
    R::Client: RuntimeClient<R> + TensorOps<R> + TypeConversionOps<R>,
{
    let source = format!(
        "{} step {} of '{}'",
        input.prefix,
        input.step,
        input.dir.display()
    );
    let mut weights =
        stage_checkpoint_tensors(&source, input.model, input.params, StageScope::Trainable)?;
    let frozen = input.frozen.host(&source, input.model, input.params)?;
    weights.extend(
        frozen
            .tensors
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.clone())),
    );
    let optimizer = stage_optimizer_state(input.optimizer, input.name_maps)?;

    let mut meta = CheckpointMeta::with_data_hash(
        input.step,
        input.model_config,
        input.data_sources,
        "",
        Some(input.position),
    );
    meta.weight_decay_rule = Some(input.weight_decay.rule);
    meta.no_decay_patterns = input.weight_decay.no_decay_patterns.clone();
    meta.loss_scale = input.loss_scale;
    meta.holdout_rows = input.eval.holdout_rows;
    meta.best_eval_loss = input.eval.best_eval_loss;
    meta.evals_since_best = input.eval.evals_since_best;
    meta.world_size = Some(input.world_size);
    meta.global_batch_rows = Some(input.global_batch_rows);
    meta.spike_guard = input.spike_guard;

    Ok(SetSnapshot::from_parts(
        input.dir.to_path_buf(),
        input.prefix.to_string(),
        input.step,
        weights,
        optimizer,
        meta,
    ))
}
