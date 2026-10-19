//! The background thread that commits checkpoint jobs in submission order.
//!
//! The writer holds at most one job. A new job waits until the job before it
//! ends, so at most one snapshot is in host memory. No job is dropped or merged.

use super::files::CheckpointLabel;
use super::job::{panic_text, CheckpointOutcome, Committer, Job};
use crate::prelude::{Error, Result};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread::JoinHandle;

/// One background thread that commits jobs in submission order.
///
/// The thread holds CPU tensors and paths only. Outcomes come back in job
/// order. [`finish`](Self::finish) and `Drop` join the thread.
pub struct CheckpointWriter {
    jobs: Option<Sender<Job>>,
    results: Receiver<Vec<CheckpointOutcome>>,
    thread: Option<JoinHandle<()>>,
    /// Step, first label and directory of the job the thread holds.
    in_flight: Option<(usize, CheckpointLabel, PathBuf)>,
    received: VecDeque<CheckpointOutcome>,
    warned: bool,
}

impl CheckpointWriter {
    /// Start the writer thread over the files committer.
    ///
    /// # Errors
    /// Returns an error when the thread does not start.
    pub fn spawn() -> Result<Self> {
        Self::spawn_with(Committer::files())
    }

    fn spawn_with(mut committer: Committer) -> Result<Self> {
        let (jobs, job_rx) = mpsc::channel::<Job>();
        let (result_tx, results) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("checkpoint-writer".to_string())
            .spawn(move || {
                for job in job_rx {
                    let outcomes = committer.run(&job);
                    // The snapshot goes before the result, so the next snapshot never meets it.
                    drop(job);
                    if result_tx.send(outcomes).is_err() {
                        break;
                    }
                }
            })
            .map_err(|e| Error::TrainingError {
                reason: format!(
                    "checkpoint writer thread did not start: {e}. Set \
                     `trainer.async_checkpoint: false` to commit on the training thread."
                ),
            })?;
        Ok(Self {
            jobs: Some(jobs),
            results,
            thread: Some(thread),
            in_flight: None,
            received: VecDeque::new(),
            warned: false,
        })
    }

    /// Move every result that arrived into the outcome queue, without blocking.
    fn collect(&mut self) {
        loop {
            match self.results.try_recv() {
                Ok(outcomes) => self.received_job(outcomes),
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    self.lose_in_flight();
                    return;
                }
            }
        }
    }

    fn received_job(&mut self, outcomes: Vec<CheckpointOutcome>) {
        self.in_flight = None;
        self.received.extend(outcomes);
    }

    /// The thread ended without a result for its job: that job failed.
    fn lose_in_flight(&mut self) {
        if let Some((step, label, dir)) = self.in_flight.take() {
            self.received.push_back(CheckpointOutcome::Failed {
                step,
                label,
                dir,
                error: "the checkpoint writer thread stopped before it finished".to_string(),
            });
        }
    }

    /// Block until the job in flight ends. Returns at once when the writer is idle.
    ///
    /// The first wait of a run prints one warning.
    pub fn wait_idle(&mut self) {
        self.collect();
        if self.in_flight.is_none() {
            return;
        }
        if !self.warned {
            self.warned = true;
            eprintln!(
                "Warning: checkpoint writer slower than save_interval. Each save waits for \
                 the commit before it. Raise `trainer.save_interval` to stop the waits."
            );
        }
        match self.results.recv() {
            Ok(outcomes) => self.received_job(outcomes),
            Err(_) => self.lose_in_flight(),
        }
    }

    /// Hand `job` to the thread. Waits first for the job in flight, if any.
    ///
    /// A job without labels commits nothing and is not sent.
    ///
    /// # Errors
    /// Returns an error when the thread is gone. The job is then not written.
    pub fn submit(&mut self, job: Job) -> Result<()> {
        self.wait_idle();
        let Some(&first) = job.labels.first() else {
            return Ok(());
        };
        let (step, dir) = (job.step, job.snapshot.dir().to_path_buf());
        let sent = self.jobs.as_ref().map(|jobs| jobs.send(job));
        if let Some(Ok(())) = sent {
            self.in_flight = Some((step, first, dir));
            return Ok(());
        }
        Err(Error::TrainingError {
            reason: format!(
                "checkpoint `{first}` of step {step} was not written: the checkpoint writer \
                 thread stopped. Set `trainer.async_checkpoint: false` to commit on the \
                 training thread."
            ),
        })
    }

    /// Every outcome that arrived and was not taken, in job order. Never blocks.
    pub fn take_outcomes(&mut self) -> Vec<CheckpointOutcome> {
        self.collect();
        self.received.drain(..).collect()
    }

    /// Close the job channel, join the thread, and queue every remaining outcome.
    ///
    /// The outcomes stay queued for [`take_outcomes`](Self::take_outcomes).
    ///
    /// # Errors
    /// Returns the first `Failed` outcome still queued, as an error, or an
    /// error when the thread panicked outside a job.
    pub fn finish(&mut self) -> Result<()> {
        if let Some(panic) = self.join() {
            return Err(Error::TrainingError {
                reason: format!(
                    "checkpoint writer thread panicked: {panic}. Check the checkpoint \
                     directory, then resume from the last committed checkpoint."
                ),
            });
        }
        match self.received.iter().find_map(CheckpointOutcome::to_error) {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Close the job channel, join the thread, and queue every remaining outcome.
    ///
    /// Returns the panic text when the thread panicked outside a job.
    fn join(&mut self) -> Option<String> {
        self.jobs = None;
        let joined = self.thread.take().map(JoinHandle::join);
        self.collect();
        self.lose_in_flight();
        match joined {
            Some(Err(payload)) => Some(panic_text(&*payload)),
            _ => None,
        }
    }
}

impl Drop for CheckpointWriter {
    /// Join the thread, and print every failure nobody took. Never panics.
    fn drop(&mut self) {
        if let Some(panic) = self.join() {
            eprintln!("Warning: checkpoint writer thread panicked: {panic}.");
        }
        for outcome in self.received.drain(..) {
            if let Some(error) = outcome.to_error() {
                eprintln!("Warning: {error}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::snapshot::SetSnapshot;
    use super::super::write::{commit_set, write_set_files};
    use super::*;
    use crate::config::{find_latest_checkpoint, CheckpointMeta, LitConfig};
    use crate::data::DataPosition;
    use crate::prelude::Tensor;
    use boostr::{CpuDevice, CpuRuntime};
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::{Arc, Mutex, Weak};

    const PREFIX: &str = "m_512_1k_f32";

    type Log = Arc<Mutex<Vec<(usize, CheckpointLabel, bool)>>>;

    /// An empty directory under the crate's `target/`.
    fn scratch(tag: &str) -> PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("unit-scratch")
            .join(format!("writer-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    fn snapshot(dir: &Path, step: usize) -> Arc<SetSnapshot> {
        let device = CpuDevice::new();
        let tensor = |v: f32| {
            Tensor::<CpuRuntime>::from_slice(&[v, v + 1.0], &[2], &device).expect("tensor")
        };
        let weights = HashMap::from([("model.norm.weight".to_string(), tensor(step as f32))]);
        let optimizer = HashMap::from([("model.norm.weight.exp_avg".to_string(), tensor(0.5))]);
        let model = LitConfig::nano_baseline().model;
        let position = Some(DataPosition::default());
        let meta = CheckpointMeta::with_data_hash(step, &model, &[], "", position);
        let (dir, prefix) = (dir.to_path_buf(), PREFIX.to_string());
        Arc::new(SetSnapshot::from_parts(
            dir, prefix, step, weights, optimizer, meta,
        ))
    }

    fn job(dir: &Path, step: usize, labels: &[CheckpointLabel], keep: usize) -> Job {
        Job {
            step,
            labels: labels.to_vec(),
            snapshot: snapshot(dir, step),
            keep,
        }
    }

    fn steps(dir: &Path, steps: &[usize]) -> Vec<Job> {
        steps
            .iter()
            .map(|&step| job(dir, step, &[CheckpointLabel::Step(step)], 0))
            .collect()
    }

    /// A committer that logs each commit, then runs `fault` on its step.
    fn logging(log: &Log, fault: impl Fn(usize) -> Result<()> + Send + 'static) -> Committer {
        let commits = Arc::clone(log);
        Committer::new(
            Box::new(
                move |snapshot: &SetSnapshot, label: CheckpointLabel, release: bool| {
                    let step = snapshot.step();
                    commits.lock().expect("log").push((step, label, release));
                    fault(step)
                },
            ),
            Box::new(|_: &SetSnapshot, _: usize| {}),
        )
    }

    fn failures(outcomes: &[CheckpointOutcome]) -> Vec<(usize, CheckpointLabel, String)> {
        outcomes
            .iter()
            .filter_map(|outcome| match outcome {
                CheckpointOutcome::Failed {
                    step, label, error, ..
                } => Some((*step, *label, error.clone())),
                CheckpointOutcome::Committed { .. } => None,
            })
            .collect()
    }

    fn committed_steps(outcomes: &[CheckpointOutcome]) -> Vec<usize> {
        outcomes
            .iter()
            .filter_map(|outcome| match outcome {
                CheckpointOutcome::Committed { step, .. } => Some(*step),
                CheckpointOutcome::Failed { .. } => None,
            })
            .collect()
    }

    #[test]
    fn jobs_commit_in_order_with_one_in_flight() {
        let dir = scratch("fifo");
        let log = Log::default();
        let mut writer = CheckpointWriter::spawn_with(logging(&log, |_| Ok(()))).expect("spawn");
        let mut previous: Option<Weak<SetSnapshot>> = None;
        let mut outcomes = Vec::new();
        for (index, job) in steps(&dir, &[1, 2, 3]).into_iter().enumerate() {
            let weak = Arc::downgrade(&job.snapshot);
            writer.submit(job).expect("submit");
            outcomes.extend(writer.take_outcomes());
            // The submit waited for the job before it, whose snapshot is gone.
            if let Some(previous) = previous.take() {
                assert!(previous.upgrade().is_none(), "two snapshots in memory");
                assert!(committed_steps(&outcomes).contains(&index), "{outcomes:?}");
            }
            previous = Some(weak);
        }
        let both = &[CheckpointLabel::Best, CheckpointLabel::Step(4)];
        writer.submit(job(&dir, 4, both, 0)).expect("submit");
        writer.finish().expect("finish");
        outcomes.extend(writer.take_outcomes());
        assert_eq!(committed_steps(&outcomes), vec![1, 2, 3, 4]);
        assert_eq!(
            outcomes.last(),
            Some(&CheckpointOutcome::Committed {
                step: 4,
                labels: both.to_vec(),
            })
        );
        let commits = log.lock().expect("log").clone();
        assert_eq!(
            commits,
            vec![
                (1, CheckpointLabel::Step(1), true),
                (2, CheckpointLabel::Step(2), true),
                (3, CheckpointLabel::Step(3), true),
                (4, CheckpointLabel::Best, false),
                (4, CheckpointLabel::Step(4), true),
            ],
            "only the last label releases the optimizer state"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn an_error_at_step_four_is_reported_with_step_four() {
        let dir = scratch("error");
        let fault = |step: usize| match step {
            4 => Err(Error::TrainingError {
                reason: "injected write error".to_string(),
            }),
            _ => Ok(()),
        };
        let mut writer =
            CheckpointWriter::spawn_with(logging(&Log::default(), fault)).expect("spawn");
        for job in steps(&dir, &[2, 4, 6]) {
            writer.submit(job).expect("submit");
        }
        let err = writer.finish().expect_err("step 4 failed").to_string();
        assert!(err.contains("`step_4` of step 4"), "{err}");
        assert!(err.contains("injected write error"), "{err}");
        assert!(err.contains(&dir.display().to_string()), "{err}");
        let outcomes = writer.take_outcomes();
        assert_eq!(committed_steps(&outcomes), vec![2, 6]);
        let failed = failures(&outcomes);
        assert_eq!(failed.len(), 1);
        assert_eq!((failed[0].0, failed[0].1), (4, CheckpointLabel::Step(4)));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_panic_becomes_a_failure_with_its_message() {
        let dir = scratch("panic");
        let fault = |step: usize| -> Result<()> {
            match step {
                1 => panic!("disk vanished at step {step}"),
                2 => std::panic::panic_any("static text panic"),
                _ => Ok(()),
            }
        };
        let mut writer =
            CheckpointWriter::spawn_with(logging(&Log::default(), fault)).expect("spawn");
        for job in steps(&dir, &[1, 2, 3]) {
            writer.submit(job).expect("submit after a panic");
        }
        assert!(writer.finish().is_err());
        let outcomes = writer.take_outcomes();
        let failed = failures(&outcomes);
        assert_eq!(failed.len(), 2, "{outcomes:?}");
        assert!(
            failed[0].2.contains("disk vanished at step 1"),
            "{failed:?}"
        );
        assert!(failed[1].2.contains("static text panic"), "{failed:?}");
        assert_eq!(
            committed_steps(&outcomes),
            vec![3],
            "the writer survives a panic"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_kill_before_the_meta_leaves_the_previous_step_latest() {
        let dir = scratch("kill");
        let committer = Committer::new(
            Box::new(
                |snapshot: &SetSnapshot, label: CheckpointLabel, release: bool| {
                    if snapshot.step() == 4 {
                        write_set_files(snapshot, label, release)?;
                        panic!("killed before the meta");
                    }
                    commit_set(snapshot, label, release)
                },
            ),
            Box::new(|_: &SetSnapshot, _: usize| {}),
        );
        let mut writer = CheckpointWriter::spawn_with(committer).expect("spawn");
        for job in steps(&dir, &[2, 4]) {
            writer.submit(job).expect("submit");
        }
        assert!(writer.finish().is_err());
        let failed = failures(&writer.take_outcomes());
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].0, 4);
        assert!(failed[0].2.contains("killed before the meta"), "{failed:?}");

        let killed = CheckpointLabel::Step(4);
        assert!(dir.join(killed.weights_file_name(PREFIX)).exists());
        assert!(dir.join(killed.optimizer_file_name(PREFIX)).exists());
        assert!(!dir.join(killed.meta_file_name()).exists());
        let (_, meta, step) = find_latest_checkpoint(&dir.to_string_lossy())
            .expect("scan")
            .expect("step 2 is committed");
        assert_eq!(step, 2);
        assert!(meta.ends_with(&CheckpointLabel::Step(2).meta_file_name()));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn drop_and_finish_join_and_commit_the_submitted_set() {
        let dir = scratch("join");
        let slow = || {
            Committer::new(
                Box::new(
                    |snapshot: &SetSnapshot, label: CheckpointLabel, release: bool| {
                        std::thread::sleep(std::time::Duration::from_millis(50));
                        commit_set(snapshot, label, release)
                    },
                ),
                Box::new(|_: &SetSnapshot, _: usize| {}),
            )
        };
        {
            let mut writer = CheckpointWriter::spawn_with(slow()).expect("spawn");
            writer.submit(steps(&dir, &[1]).remove(0)).expect("submit");
        }
        assert!(dir.join(CheckpointLabel::Step(1).meta_file_name()).exists());

        let mut writer = CheckpointWriter::spawn_with(slow()).expect("spawn");
        writer.submit(steps(&dir, &[2]).remove(0)).expect("submit");
        writer.finish().expect("finish");
        assert_eq!(committed_steps(&writer.take_outcomes()), vec![2]);
        let meta = dir.join(CheckpointLabel::Step(2).meta_file_name());
        let meta = CheckpointMeta::load(&meta.to_string_lossy()).expect("meta");
        assert_eq!(meta.step, 2);
        assert_eq!(
            meta.optimizer_file.as_deref(),
            Some(
                CheckpointLabel::Step(2)
                    .optimizer_file_name(PREFIX)
                    .as_str()
            )
        );
        assert!(meta
            .checkpoint_file
            .ends_with(&CheckpointLabel::Step(2).weights_file_name(PREFIX)));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn cleanup_runs_only_after_a_committed_step_label() {
        let dir = scratch("cleanup");
        let pruned = Arc::new(Mutex::new(Vec::new()));
        let prunes = Arc::clone(&pruned);
        let mut committer = Committer::new(
            Box::new(
                |snapshot: &SetSnapshot, _: CheckpointLabel, _: bool| match snapshot.step() {
                    3 => Err(Error::TrainingError {
                        reason: "injected".to_string(),
                    }),
                    _ => Ok(()),
                },
            ),
            Box::new(move |snapshot: &SetSnapshot, keep: usize| {
                prunes.lock().expect("log").push((snapshot.step(), keep));
            }),
        );
        let (best, step) = (CheckpointLabel::Best, CheckpointLabel::Step);
        for job in [
            job(&dir, 1, &[best], 2),
            job(&dir, 2, &[step(2)], 2),
            job(&dir, 3, &[step(3)], 2),
            job(&dir, 4, &[step(4)], 0),
            job(&dir, 5, &[best, step(5)], 2),
        ] {
            committer.run(&job);
        }
        assert_eq!(*pruned.lock().expect("log"), vec![(2, 2), (5, 2)]);
        let _ = std::fs::remove_dir_all(dir);
    }
}
