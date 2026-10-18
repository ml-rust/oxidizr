//! The data-parallel hooks of `LitTrainer`: its communicator, its abort hook,
//! the error-safe step, and the rank-0 broadcasts.

use super::broadcast::{broadcast_model_parameters, broadcast_stop_flag};
use crate::data::DataLoader;
use crate::model::LitGPT;
use crate::prelude::{DType, Error, Result};
use crate::trainer::LitTrainer;
use boostr::ops::{FusedOptimizerOps, TensorOps};
use boostr::runtime::{Communicator, Runtime, RuntimeClient};
use std::sync::Arc;

/// Aborts the communicator group of one rank, so peers blocked in a collective return.
///
/// `Communicator` has no abort call, so the launcher passes the backend's own.
/// For `LocalCommunicator` it calls `LocalCommunicator::abort`. For NCCL it
/// calls `NcclCommunicator::abort` on every rank. See [`super::RankGroup`].
pub type AbortHook = Arc<dyn Fn() + Send + Sync>;

/// The early-stop reason of a rank other than 0, set by the step-boundary broadcast.
pub const PEER_STOP_REASON: &str = "rank 0 stopped early";

impl<'a, R: Runtime<DType = DType>> LitTrainer<'a, R> {
    /// Make this trainer one rank of the data-parallel group `comm`.
    ///
    /// The default is `NoOpCommunicator`: one rank and no collective.
    /// With more than one rank, each step sums its gradients over `comm`.
    /// Only rank 0 writes files, evaluates, and prints diagnostics.
    ///
    /// The row layout must come from a loader built with
    /// `LitDataLoader::with_rank(rank, world)`, so steps count global batches.
    ///
    /// # Errors
    /// Returns an error when the rank of `comm` is not below its world size,
    /// or its world size differs from the row layout's.
    pub fn with_communicator(mut self, comm: Arc<dyn Communicator>) -> Result<Self> {
        let (rank, world) = (comm.rank(), comm.world_size());
        if rank >= world {
            return Err(Error::TrainingError {
                reason: format!(
                    "communicator rank {rank} is not below its world size {world}. Pass a \
                     communicator of a valid group"
                ),
            });
        }
        let layout_world = self.row_layout.world_size();
        if world != layout_world {
            return Err(Error::TrainingError {
                reason: format!(
                    "communicator world size {world} differs from the row layout's \
                     {layout_world}. Build the loader with LitDataLoader::with_rank(rank, \
                     {world}) and take its row layout"
                ),
            });
        }
        self.comm = comm;
        Ok(self)
    }

    /// Call `hook` when a step or the fit loop of this rank fails in a world above 1.
    ///
    /// Without a hook, peers of a failed rank wait until the backend's timeout.
    pub fn with_abort_hook(mut self, hook: AbortHook) -> Self {
        self.abort_hook = Some(hook);
        self
    }

    /// Ranks of the data-parallel group. 1 without data parallelism.
    pub fn world_size(&self) -> usize {
        self.comm.world_size()
    }

    /// This trainer's rank, below [`world_size`](Self::world_size).
    pub fn rank(&self) -> usize {
        self.comm.rank()
    }

    /// True on ranks other than 0. A quiet rank prints no diagnostic and writes no file.
    pub(crate) fn quiet(&self) -> bool {
        self.rank() != 0
    }

    /// Abort the communicator group through the abort hook.
    ///
    /// Does nothing in a world of 1 or without a hook.
    /// After an abort, every later collective of the group returns an error.
    pub fn abort_collectives(&self) {
        if self.world_size() <= 1 {
            return;
        }
        if let Some(hook) = &self.abort_hook {
            hook();
        }
    }

    /// Block until every rank arrives. Does nothing in a world of 1.
    ///
    /// # Errors
    /// Returns an error when the barrier fails, for example after an abort.
    pub(crate) fn barrier(&self) -> Result<()> {
        if self.world_size() <= 1 {
            return Ok(());
        }
        self.comm.barrier().map_err(|e| Error::DistributedError {
            reason: format!("rank {} barrier failed: {e}", self.rank()),
        })
    }
}

impl<'a, R> LitTrainer<'a, R>
where
    R: Runtime<DType = DType>,
    R::Client: RuntimeClient<R>
        + TensorOps<R>
        + FusedOptimizerOps<R>
        + boostr::ScalarOps<R>
        + boostr::BinaryOps<R>
        + boostr::CompareOps<R>
        + boostr::ConvOps<R>
        + boostr::FlashAttentionOps<R>
        + boostr::IndexingOps<R>
        + boostr::NormalizationOps<R>
        + boostr::QuantMatmulOps<R>
        + boostr::RoPEOps<R>
        + boostr::TypeConversionOps<R>
        + boostr::DequantOps<R>,
{
    /// Performs one training step (including gradient accumulation)
    /// Returns (loss, grad_norm)
    ///
    /// # Gradient Clipping Strategy
    /// Gradients are clipped in place. `crate::grad_clip::clip_gradients` first
    /// clamps each element to `config.max_grad_value` when that is set. It then
    /// clips to `config.max_grad_norm`. The step then clips, optionally, to the
    /// tighter `config.target_grad_norm`. A step whose gradient norm is inf/NaN
    /// is skipped, not clipped.
    ///
    /// The clips measure the norm over `self.parameter_ids`: the tensors the
    /// optimizer steps, and the only keys the merged grad store holds.
    /// `backward_wrt` still retains the loss node's `dL/dL = 1` seed and every
    /// activation on a path to a trainable parameter, but that per-micro-batch
    /// store is filtered to `parameter_ids` on the way into the merged store
    /// and then dropped.
    ///
    /// # Gradient Accumulation Strategy
    /// Each micro-batch runs its own forward AND backward, and its parameter
    /// gradients are summed into one store. By linearity,
    /// `grad(sum_k scaled_loss_k) == sum_k grad(scaled_loss_k)`, so this
    /// computes what chaining the micro-batch losses into one graph and running
    /// a single backward computed — but only one micro-batch's activations are
    /// alive at a time, instead of `gradient_accumulation` of them.
    ///
    /// Float addition is not associative, so the accumulation ORDER matters:
    /// gradients land in micro-batch order rather than in the merged
    /// graph's reverse-topological order. A parameter reached from several
    /// micro-batches can therefore differ in its last bits, and the reported
    /// `grad_norm` with it. The reported loss cannot move: it is the same
    /// device `add` over the same scalars in the same micro-batch order.
    ///
    /// # Loss Weighting
    /// The step draws all its micro-batches before the first forward pass.
    /// [`StepWeighting`](crate::step_loss::StepWeighting) then picks the
    /// cross-entropy form. A masked step
    /// weights every scored token of the step by `1 / total`. An unmasked step
    /// keeps the per-micro-batch mean. Either way the cross-entropy part of
    /// the reported loss is the per-token mean over the step.
    ///
    /// # Loss Scaling
    /// An f16 run multiplies each backward loss by the loss scale on top of
    /// that weighting. After accumulation, every gradient is widened to f32
    /// and divided by the scale, before any clip. A non-finite norm skips the
    /// optimizer step, so its timestep stays put, and the scale backs off.
    /// The step still draws its batches. The reported loss and norm are unscaled.
    ///
    /// # Data Parallelism
    /// With `W` ranks the step has `M = gradient_accumulation * W` micro-batches.
    /// Each micro-batch loss is scaled by `1 / M`. Every step first sums its
    /// scored count and its mask flag over every rank, so `total` is the global
    /// count, and a mask on some ranks only is an error on every rank. After the
    /// local backward passes, one flat sum adds every gradient and the logged
    /// loss over all ranks. Unscale, clips, the overflow decision and the
    /// optimizer step then run on the summed gradients, which are bit-identical
    /// on every rank. So every rank takes the same skip and clip decisions, and
    /// the replicas stay bit-equal. Every rank returns the global loss.
    ///
    /// At world 1 no collective runs and the step keeps its bits.
    ///
    /// # Errors
    /// Returns the first error of the step. In a world above 1 the step calls
    /// [`abort_collectives`](Self::abort_collectives) first, so peers blocked in
    /// a collective return an error instead of waiting.
    pub fn training_step<D: DataLoader<R>>(
        &mut self,
        model: &LitGPT<R>,
        data_loader: &mut D,
    ) -> Result<(f64, f64)> {
        let result = self.step_once(model, data_loader);
        if result.is_err() {
            self.abort_collectives();
        }
        result
    }

    /// Copy rank 0's model tensors into every rank, trainable and frozen.
    ///
    /// `trainer.seed` is optional, so ranks can build different initial
    /// weights. A resumed run can also differ by rank in what it loaded.
    /// Data parallelism needs bit-equal replicas, so rank 0's tensors win.
    /// `fit` calls this before its first step. A caller that runs
    /// `training_step` directly must call it once before the first step.
    ///
    /// Does nothing in a world of 1.
    ///
    /// # Errors
    /// Returns the broadcast error, after [`abort_collectives`](Self::abort_collectives).
    pub fn sync_initial_parameters(&mut self, model: &LitGPT<R>) -> Result<()> {
        let result =
            broadcast_model_parameters(&*self.comm, &self.client, model, &*self.params).map(drop);
        if result.is_err() {
            self.abort_collectives();
        }
        result
    }

    /// Agree on rank 0's early-stop decision at a step boundary.
    ///
    /// Only rank 0 evaluates. Every other rank learns the decision here and
    /// stops at the same boundary. Does nothing in a world of 1.
    ///
    /// # Errors
    /// Returns an error when the broadcast fails.
    pub(crate) fn sync_early_stop(&mut self) -> Result<()> {
        if self.world_size() <= 1 {
            return Ok(());
        }
        let stop = !self.quiet() && self.early_stop_reason().is_some();
        let agreed = broadcast_stop_flag::<R, _>(&*self.comm, &self.client, stop)?;
        if agreed && self.quiet() {
            self.peer_stop = true;
        }
        Ok(())
    }
}
