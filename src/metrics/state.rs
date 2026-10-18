//! The metrics state a trainer carries between steps.

use super::config::{MetricsConfig, SpikeAction};
use super::scalars::ScalarLog;
use super::spike::{SpikeEvent, SpikeGuard, SpikeGuardState};
use crate::config::TrainerConfig;
use crate::prelude::Result;
use std::sync::Arc;

/// Maps `(0-based step index, step loss)` to the loss the step reports and the guard judges.
///
/// A fault-injection hook for tests: it scripts a loss spike at a known step.
/// Every data-parallel rank must get the same probe, or the ranks diverge.
pub type LossProbe = Arc<dyn Fn(usize, f64) -> f64 + Send + Sync>;

/// What `run.json` records about throughput and spikes.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RunMetrics {
    /// Session tokens per second at the last step. `None` when no step ran.
    pub tokens_per_sec: Option<f64>,
    /// MFU at the last step. `None` without `trainer.metrics.peak_tflops`.
    pub mfu: Option<f64>,
    /// The guard's counters. `None` without `trainer.spike_guard`.
    pub spikes: Option<SpikeGuardState>,
}

/// The metrics state of one trainer.
pub struct TrainMetrics {
    config: MetricsConfig,
    spike: Option<SpikeGuard>,
    probe: Option<LossProbe>,
    /// Global scored tokens of the last step. `None` for an unmasked step.
    pub(crate) last_scored: Option<usize>,
    /// The guard's event of the last step.
    pub(crate) last_spike: Option<SpikeEvent>,
    /// Session tokens per second and MFU at the last reported step.
    pub(crate) last_rates: Option<(f64, Option<f64>)>,
    pub(crate) scalars: ScalarLog,
}

impl TrainMetrics {
    /// The metrics state of `config`, with an empty spike EMA.
    ///
    /// # Errors
    /// Returns an error when `trainer.metrics` or `trainer.spike_guard` holds a bad value.
    pub fn new(config: &TrainerConfig) -> Result<Self> {
        let metrics = config.metrics.unwrap_or_default();
        metrics.check()?;
        let spike = config.spike_guard.map(SpikeGuard::new).transpose()?;
        Ok(Self {
            config: metrics,
            spike,
            probe: None,
            last_scored: None,
            last_spike: None,
            last_rates: None,
            scalars: ScalarLog::off(),
        })
    }

    pub fn peak_tflops(&self) -> Option<f64> {
        self.config.peak_tflops
    }

    pub(crate) fn set_probe(&mut self, probe: LossProbe) {
        self.probe = Some(probe);
    }

    /// The step loss after the probe, when one is set.
    pub(crate) fn probe_loss(&self, index: usize, loss: f64) -> f64 {
        match &self.probe {
            Some(probe) => probe(index, loss),
            None => loss,
        }
    }

    /// Judges the loss of the 0-based step `index` and returns true when the
    /// step must skip the optimizer. Without a guard it never skips.
    pub(crate) fn judge(&mut self, index: usize, loss: f64) -> bool {
        self.last_spike = self
            .spike
            .as_mut()
            .and_then(|guard| guard.observe(index + 1, loss));
        self.last_spike
            .is_some_and(|e| e.action == SpikeAction::Skip)
    }

    /// The guard's state for a checkpoint meta. `None` without a guard.
    pub fn spike_state(&self) -> Option<SpikeGuardState> {
        self.spike.as_ref().map(SpikeGuard::state)
    }

    pub(crate) fn spike_guard_mut(&mut self) -> Option<&mut SpikeGuard> {
        self.spike.as_mut()
    }

    pub fn run_metrics(&self) -> RunMetrics {
        RunMetrics {
            tokens_per_sec: self.last_rates.map(|(rate, _)| rate),
            mfu: self.last_rates.and_then(|(_, mfu)| mfu),
            spikes: self.spike_state(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LitConfig;
    use crate::metrics::SpikeGuardConfig;

    #[test]
    fn without_a_guard_nothing_skips_and_nothing_is_recorded() {
        let config = LitConfig::nano_baseline().trainer;
        let mut metrics = TrainMetrics::new(&config).expect("metrics");
        assert!(!metrics.judge(100, 1e9));
        assert_eq!(metrics.spike_state(), None);
        assert_eq!(metrics.peak_tflops(), None);
        assert_eq!(metrics.run_metrics(), RunMetrics::default());
    }

    #[test]
    fn the_probe_rewrites_the_loss_and_the_guard_skips_its_spike() {
        let mut config = LitConfig::nano_baseline().trainer;
        config.spike_guard = Some(SpikeGuardConfig {
            window: 4,
            z_threshold: 6.0,
            min_steps: 2,
            action: SpikeAction::Skip,
        });
        let mut metrics = TrainMetrics::new(&config).expect("metrics");
        metrics.set_probe(Arc::new(
            |index: usize, loss: f64| {
                if index == 3 {
                    1e3
                } else {
                    loss
                }
            },
        ));
        let losses: Vec<f64> = (0..5)
            .map(|i| metrics.probe_loss(i, 1.0 + 0.1 * (i % 2) as f64))
            .collect();
        let skips: Vec<bool> = losses
            .iter()
            .enumerate()
            .map(|(i, &loss)| metrics.judge(i, loss))
            .collect();
        assert_eq!(skips, [false, false, false, true, false]);
        let state = metrics.spike_state().expect("a guard");
        assert_eq!((state.observed, state.detected, state.skipped), (4, 1, 1));
    }
}
