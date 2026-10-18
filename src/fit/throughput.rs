//! Step timing of one fit session: the session rate and a rolling recent rate.

use std::collections::VecDeque;
use std::time::Instant;

/// Steps in the rolling mean of the recent step time.
const RECENT_STEPS: usize = 5;

/// Clocks of one fit session.
pub(crate) struct Throughput {
    samples_per_step: usize,
    session_start: Instant,
    step_start: Instant,
    recent: VecDeque<f64>,
}

impl Throughput {
    /// Starts the session clock. Each step processes `samples_per_step` samples.
    pub(crate) fn start(samples_per_step: usize) -> Self {
        let now = Instant::now();
        Self {
            samples_per_step,
            session_start: now,
            step_start: now,
            recent: VecDeque::with_capacity(RECENT_STEPS + 1),
        }
    }

    /// Samples each step processes.
    pub(crate) fn samples_per_step(&self) -> usize {
        self.samples_per_step
    }

    /// Restarts the step clock. The fit loop calls this right before its first step.
    pub(crate) fn restart_step_clock(&mut self) {
        self.step_start = Instant::now();
    }

    /// Ends the current step and starts the next one.
    ///
    /// A step's time runs from the end of the previous step, so it includes
    /// the previous step's output and checkpoint save.
    pub(crate) fn end_step(&mut self) {
        push_recent(&mut self.recent, self.step_start.elapsed().as_secs_f64());
        self.step_start = Instant::now();
    }

    /// Samples of `session_steps` steps divided by the session's elapsed seconds.
    pub(crate) fn session_rate(&self, session_steps: usize) -> f64 {
        let samples = session_steps * self.samples_per_step;
        samples as f64 / self.session_start.elapsed().as_secs_f64()
    }

    /// Mean seconds of the recent steps. 0 before the first step ends.
    pub(crate) fn mean_step_secs(&self) -> f64 {
        mean(&self.recent)
    }

    /// Samples per step divided by [`Self::mean_step_secs`]. 0 when that mean is 0.
    pub(crate) fn recent_rate(&self) -> f64 {
        rate(self.samples_per_step, self.mean_step_secs())
    }
}

fn push_recent(recent: &mut VecDeque<f64>, secs: f64) {
    recent.push_back(secs);
    if recent.len() > RECENT_STEPS {
        recent.pop_front();
    }
}

fn mean(recent: &VecDeque<f64>) -> f64 {
    if recent.is_empty() {
        return 0.0;
    }
    recent.iter().sum::<f64>() / recent.len() as f64
}

fn rate(samples_per_step: usize, mean_step_secs: f64) -> f64 {
    if mean_step_secs > 0.0 {
        samples_per_step as f64 / mean_step_secs
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_recent_window_keeps_the_last_five_steps() {
        let mut recent = VecDeque::new();
        for secs in [9.0, 1.0, 1.0, 2.0, 2.0, 4.0] {
            push_recent(&mut recent, secs);
        }
        assert_eq!(recent.len(), RECENT_STEPS);
        assert_eq!(mean(&recent), 2.0);
    }

    #[test]
    fn the_recent_rate_is_zero_without_a_step_time() {
        assert_eq!(rate(8, 0.0), 0.0);
        assert_eq!(rate(8, 2.0), 4.0);
        assert_eq!(mean(&VecDeque::new()), 0.0);
    }
}
