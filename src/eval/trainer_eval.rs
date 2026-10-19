//! The eval hooks of `LitTrainer`: attach an eval, run it at step boundaries, and report it.

use super::resume::EvalCheckpoint;
use super::run::{evaluate, EvalTotals};
use super::state::{EvalOutcome, Evaluator};
use super::stopper::Decision;
use crate::dp::PEER_STOP_REASON;
use crate::fit::{EvalReport, StepSink};
use crate::model::LitGPT;
use crate::prelude::{DType, Result};
use crate::run_record::EVALS_FILE;
use crate::trainer::LitTrainer;
use boostr::ops::{FusedOptimizerOps, TensorOps};
use boostr::runtime::{Runtime, RuntimeClient};
use std::path::Path;

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
    /// Turns on the evals of `evaluator` for this trainer's fit loop.
    ///
    /// Eval lines go to `evals.json` beside this trainer's `steps.json`.
    pub fn attach_eval(&mut self, mut evaluator: Evaluator<R>) {
        evaluator.set_evals_path(Path::new(&self.steps_json_path).with_file_name(EVALS_FILE));
        self.eval = Some(evaluator);
    }

    /// Scores the eval batches under the current weights. `None` without an eval.
    ///
    /// The pass changes no stop counter and writes no file.
    pub fn evaluate_now(&mut self, model: &LitGPT<R>) -> Result<Option<EvalTotals>> {
        let Some(eval) = self.eval.as_mut() else {
            return Ok(None);
        };
        evaluate(
            &self.client,
            model,
            &*self.params,
            &mut eval.loader,
            eval.batches,
        )
        .map(Some)
    }

    /// Runs the eval due once `completed` of `total_steps` steps are done.
    ///
    /// The eval goes to `evals.json` and to `sink`. Returns true when its loss
    /// is a new best. The fit loop then saves the `best` checkpoint, and any
    /// interval checkpoint of the same step, from one snapshot. Their metas
    /// hold this eval's counters.
    ///
    /// Only data-parallel rank 0 evaluates. The fit loop broadcasts its stop decision.
    pub(crate) fn eval_if_due(
        &mut self,
        model: &LitGPT<R>,
        completed: usize,
        total_steps: usize,
        sink: &mut dyn StepSink,
    ) -> Result<bool> {
        if self.quiet() {
            return Ok(false);
        }
        let due = self
            .eval
            .as_ref()
            .is_some_and(|eval| eval.is_due(completed, total_steps));
        if !due {
            return Ok(false);
        }
        let Some(totals) = self.evaluate_now(model)? else {
            return Ok(false);
        };
        let Some(eval) = self.eval.as_mut() else {
            return Ok(false);
        };
        let loss = totals.loss();
        let decision = eval.stopper.observe(loss);
        eval.evals_run += 1;
        if matches!(decision, Decision::Stop { .. }) {
            eval.stop_reason = Some(eval.stop_text());
        }
        let report = EvalReport {
            loss,
            perplexity: totals.perplexity(),
            batches: totals.batches,
            tokens: totals.tokens,
            best_loss: eval.stopper.best(),
            since_best: eval.stopper.since_best(),
        };
        eval.record(completed, &report);
        self.metrics.scalars.eval(completed, &report);
        sink.on_eval(completed, &report)?;
        Ok(decision == Decision::Improved)
    }

    /// Why the run stops early. `None` without an eval, or while the patience lasts.
    ///
    /// A data-parallel rank other than 0 reports [`PEER_STOP_REASON`] once
    /// rank 0's stop decision reached it.
    pub fn early_stop_reason(&self) -> Option<&str> {
        self.eval
            .as_ref()
            .and_then(Evaluator::stop_reason)
            .or_else(|| self.peer_stop.then_some(PEER_STOP_REASON))
    }

    /// What the evals reached. `None` without an eval.
    pub fn eval_outcome(&self) -> Option<EvalOutcome> {
        self.eval.as_ref().map(Evaluator::outcome)
    }

    /// The eval keys of a checkpoint saved now.
    pub(crate) fn eval_checkpoint(&self) -> EvalCheckpoint {
        self.eval
            .as_ref()
            .map(Evaluator::checkpoint_state)
            .unwrap_or_default()
    }
}
