//! The metrics hooks of `LitTrainer`: the loss probe, the spike guard state, the run figures.

use super::spike::SpikeGuardState;
use super::state::{LossProbe, RunMetrics};
use crate::prelude::{DType, Result};
use crate::trainer::LitTrainer;
use boostr::runtime::Runtime;

impl<'a, R: Runtime<DType = DType>> LitTrainer<'a, R> {
    /// Rewrite every step loss through `probe` before the report and the spike guard.
    ///
    /// A fault-injection hook for tests. Give every data-parallel rank the same probe.
    pub fn with_loss_probe(mut self, probe: LossProbe) -> Self {
        self.metrics.set_probe(probe);
        self
    }

    /// Continue the spike guard from a resumed checkpoint's `saved` state.
    ///
    /// - Guard and state: the guard continues from `saved`.
    /// - A guard without a state at a step past 0: the EMA restarts empty, with a warning.
    /// - A state without a guard: the state is dropped, with a warning.
    ///
    /// Only rank 0 prints the warnings.
    ///
    /// # Errors
    /// Returns an error when `saved` holds a non-finite mean or a bad variance.
    pub fn restore_spike_guard(&mut self, saved: Option<SpikeGuardState>) -> Result<()> {
        let quiet = self.quiet();
        let start_step = self.start_step;
        match (self.metrics.spike_guard_mut(), saved) {
            (Some(guard), Some(state)) => guard.restore(state),
            (Some(_), None) if start_step > 0 && !quiet => {
                eprintln!(
                    "WARNING: the checkpoint records no spike guard state. The guard restarts \
                     its loss average and flags nothing until it holds min_steps losses."
                );
                Ok(())
            }
            (None, Some(state)) if !quiet => {
                eprintln!(
                    "WARNING: the checkpoint records a spike guard state ({} spikes skipped), \
                     but this run sets no trainer.spike_guard. The state is dropped.",
                    state.skipped
                );
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// The spike guard's EMA and counters. `None` without `trainer.spike_guard`.
    pub fn spike_guard_state(&self) -> Option<SpikeGuardState> {
        self.metrics.spike_state()
    }

    /// Throughput and spike figures for `run.json`.
    pub fn run_metrics(&self) -> RunMetrics {
        self.metrics.run_metrics()
    }
}
