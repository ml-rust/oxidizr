//! The checkpoint saves of the fit loop: snapshot on the training thread, then commit.
//!
//! Without `trainer.async_checkpoint` the commit runs inline. With it, the
//! [`CheckpointWriter`] thread commits while the next steps run. Both modes
//! run the same snapshot and commit code, so their weights files are byte-equal.

use super::sink::{CheckpointNote, StepSink};
use crate::checkpoint_set::{CheckpointLabel, CheckpointOutcome, CheckpointWriter, Committer, Job};
use crate::model::LitGPT;
use crate::prelude::{DType, Result};
use crate::trainer::LitTrainer;
use boostr::ops::{FusedOptimizerOps, TensorOps};
use boostr::runtime::{Runtime, RuntimeClient};
use std::collections::VecDeque;
use std::sync::Arc;

/// The background writer of a fit loop, and the outcomes it has not reported yet.
///
/// Only the rank that writes files holds a writer. No collective reads this state.
#[derive(Default)]
pub struct CheckpointQueue {
    writer: Option<CheckpointWriter>,
    pending: VecDeque<CheckpointOutcome>,
}

impl CheckpointQueue {
    /// Move every outcome the writer has into `pending`. Never blocks.
    fn collect(&mut self) {
        if let Some(writer) = self.writer.as_mut() {
            self.pending.extend(writer.take_outcomes());
        }
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
        + boostr::DequantOps<R>
        + boostr::GatedDeltaNetOps<R>,
{
    /// Start the background writer when this run commits asynchronously.
    ///
    /// A writer exists only under `trainer.async_checkpoint` with
    /// `save_interval` above 0, and only on the rank that writes files.
    pub(crate) fn start_checkpoint_writer(&mut self) -> Result<()> {
        let wanted = self.config.async_checkpoint && self.config.save_interval > 0;
        if wanted && !self.quiet() && self.checkpoints.writer.is_none() {
            self.checkpoints.writer = Some(CheckpointWriter::spawn()?);
        }
        Ok(())
    }

    /// The labels saved once `completed` steps are done, in commit order.
    ///
    /// `best` comes first when the eval of this step `improved` and
    /// `save_interval` is above 0. The interval set follows when it is due.
    pub(super) fn due_labels(&self, completed: usize, improved: bool) -> Vec<CheckpointLabel> {
        let interval = self.config.save_interval;
        let mut labels = Vec::with_capacity(2);
        if improved && interval > 0 {
            labels.push(CheckpointLabel::Best);
        }
        if interval > 0 && completed.is_multiple_of(interval) {
            labels.push(CheckpointLabel::Step(completed));
        }
        labels
    }

    /// Save `labels` of `step` from one snapshot of the current state.
    ///
    /// With a writer, this first waits for the job in flight and reports its
    /// outcome to `sink`. So each note fires before a later job can replace
    /// or prune its set. The new job then goes to the writer. Without a
    /// writer, the commit runs here. A rank that writes no files returns at once.
    ///
    /// # Errors
    /// Returns the failure of the job in flight, before any new snapshot.
    pub(crate) fn request_save(
        &mut self,
        model: &LitGPT<R>,
        step: usize,
        labels: Vec<CheckpointLabel>,
        sink: &mut dyn StepSink,
    ) -> Result<()> {
        if self.quiet() || labels.is_empty() {
            return Ok(());
        }
        if let Some(writer) = self.checkpoints.writer.as_mut() {
            writer.wait_idle();
        }
        self.drain_checkpoints(sink)?;
        let job = Job {
            step,
            labels,
            snapshot: Arc::new(self.snapshot_set(model, step)?),
            keep: self.config.max_checkpoints,
        };
        match self.checkpoints.writer.as_mut() {
            Some(writer) => writer.submit(job),
            None => {
                let outcomes = Committer::files().run(&job);
                self.checkpoints.pending.extend(outcomes);
                Ok(())
            }
        }
    }

    /// Report every committed checkpoint to `sink`, in commit order. Never blocks.
    ///
    /// `Saved` and `Best` notes fire here, on the training thread, once the
    /// meta is on disk. Under async commits a note can arrive steps later.
    ///
    /// # Errors
    /// Returns the first failed commit, naming its label, step and directory.
    pub(crate) fn drain_checkpoints(&mut self, sink: &mut dyn StepSink) -> Result<()> {
        self.checkpoints.collect();
        while let Some(outcome) = self.checkpoints.pending.pop_front() {
            if let Some(error) = outcome.to_error() {
                return Err(error);
            }
            let CheckpointOutcome::Committed { step, labels } = outcome else {
                continue;
            };
            for label in labels {
                let note = match label {
                    CheckpointLabel::Best => CheckpointNote::Best { step },
                    CheckpointLabel::Step(saved) => CheckpointNote::Saved { step: saved },
                    CheckpointLabel::Final => continue,
                };
                sink.on_checkpoint(&note)?;
            }
        }
        Ok(())
    }

    /// Wait for every submitted job, then report its outcome to `sink`.
    ///
    /// The fit loop calls this before the final save, so every `Saved` note
    /// arrives before the `Final` note.
    ///
    /// # Errors
    /// Returns the first failed commit, as [`drain_checkpoints`](Self::drain_checkpoints) does.
    pub(crate) fn finish_checkpoints(&mut self, sink: &mut dyn StepSink) -> Result<()> {
        let joined = self.join_writer();
        self.drain_checkpoints(sink)?;
        joined
    }

    /// Wait for every submitted job after an error, and print each failed commit.
    ///
    /// The run already failed, so nothing reaches the sink.
    pub(crate) fn abandon_checkpoints(&mut self) {
        let joined = self.join_writer();
        let failures: Vec<_> = self
            .checkpoints
            .pending
            .drain(..)
            .filter_map(|outcome| outcome.to_error())
            .collect();
        // A failed commit is also the join error. Print it once.
        if let (Err(e), true) = (joined, failures.is_empty()) {
            eprintln!("Error: {e}");
        }
        for error in failures {
            eprintln!("Error: {error}");
        }
    }

    /// Join the writer and queue its outcomes. Returns its thread error, if any.
    fn join_writer(&mut self) -> Result<()> {
        let Some(mut writer) = self.checkpoints.writer.take() else {
            return Ok(());
        };
        let joined = writer.finish();
        self.checkpoints.pending.extend(writer.take_outcomes());
        joined
    }
}
