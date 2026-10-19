//! One checkpoint job and the committer that writes its labels in order.

use super::cleanup::remove_old_checkpoints;
use super::files::CheckpointLabel;
use super::snapshot::SetSnapshot;
use super::write::commit_set;
use crate::prelude::{Error, Result};
use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Arc;

/// One save: the labels of step `step`, all written from one snapshot.
pub struct Job {
    /// Completed optimizer steps of the snapshot.
    pub step: usize,
    /// Labels in commit order, for example `[Best, Step(N)]`.
    pub labels: Vec<CheckpointLabel>,
    /// The host copy every label commits from.
    pub snapshot: Arc<SetSnapshot>,
    /// `max_checkpoints`: interval sets to keep after a `Step` commit. 0 keeps every set.
    pub keep: usize,
}

/// The result of one job, or of one label of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointOutcome {
    /// The meta of each of `labels` is on disk.
    Committed {
        step: usize,
        labels: Vec<CheckpointLabel>,
    },
    /// The commit of `label` returned `error` or panicked. Later labels of the job did not run.
    Failed {
        step: usize,
        label: CheckpointLabel,
        dir: PathBuf,
        error: String,
    },
}

impl CheckpointOutcome {
    /// The error of a `Failed` outcome. `None` for `Committed`.
    pub fn to_error(&self) -> Option<Error> {
        match self {
            Self::Committed { .. } => None,
            Self::Failed {
                step,
                label,
                dir,
                error,
            } => Some(Error::TrainingError {
                reason: format!(
                    "checkpoint `{label}` of step {step} was not written: {error}. Check that \
                     `{}` is writable and has free space.",
                    dir.display()
                ),
            }),
        }
    }
}

/// Writes one label of a snapshot. The flag releases the optimizer state.
type CommitFn = Box<dyn FnMut(&SetSnapshot, CheckpointLabel, bool) -> Result<()> + Send>;
/// Removes old interval sets: the snapshot names the directory, then the keep count.
type PruneFn = Box<dyn FnMut(&SetSnapshot, usize) + Send>;

/// Runs jobs: commits each label, then prunes after a `Step` commit.
pub struct Committer {
    commit: CommitFn,
    prune: PruneFn,
}

impl Committer {
    /// The committer of real files: [`commit_set`] and [`remove_old_checkpoints`].
    pub fn files() -> Self {
        Self::new(
            Box::new(commit_set),
            Box::new(|snapshot: &SetSnapshot, keep: usize| {
                remove_old_checkpoints(snapshot.dir(), keep)
            }),
        )
    }

    /// A committer over `commit` and `prune`, so a test can inject faults.
    pub(crate) fn new(commit: CommitFn, prune: PruneFn) -> Self {
        Self { commit, prune }
    }

    /// Commit every label of `job` in order. A panic becomes a `Failed` outcome.
    ///
    /// The last label releases the optimizer state. A `Step` commit prunes
    /// old interval sets when `keep` is above 0. A failed label stops the job.
    pub fn run(&mut self, job: &Job) -> Vec<CheckpointOutcome> {
        let mut committed = Vec::with_capacity(job.labels.len());
        let last = job.labels.len().saturating_sub(1);
        let snapshot = job.snapshot.as_ref();
        for (index, &label) in job.labels.iter().enumerate() {
            let commit = &mut self.commit;
            let result = catch_unwind(AssertUnwindSafe(|| commit(snapshot, label, index == last)));
            let error = match result {
                Ok(Ok(())) => None,
                Ok(Err(e)) => Some(e.to_string()),
                Err(payload) => Some(format!("the commit panicked: {}", panic_text(&*payload))),
            };
            if let Some(error) = error {
                let mut outcomes = Vec::with_capacity(2);
                if !committed.is_empty() {
                    outcomes.push(CheckpointOutcome::Committed {
                        step: job.step,
                        labels: committed,
                    });
                }
                outcomes.push(CheckpointOutcome::Failed {
                    step: job.step,
                    label,
                    dir: snapshot.dir().to_path_buf(),
                    error,
                });
                return outcomes;
            }
            committed.push(label);
            if matches!(label, CheckpointLabel::Step(_)) && job.keep > 0 {
                let prune = &mut self.prune;
                if let Err(payload) = catch_unwind(AssertUnwindSafe(|| prune(snapshot, job.keep))) {
                    eprintln!(
                        "Warning: removal of old checkpoints after step {} panicked: {}. \
                         Remove old sets by hand.",
                        job.step,
                        panic_text(&*payload)
                    );
                }
            }
        }
        vec![CheckpointOutcome::Committed {
            step: job.step,
            labels: committed,
        }]
    }
}

/// The text of a panic payload.
pub(super) fn panic_text(payload: &(dyn Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "a panic without a text payload".to_string()
    }
}
