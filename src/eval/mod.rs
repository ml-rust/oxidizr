//! Validation evals on held-out rows or an eval file, and early stopping.

mod config;
mod holdout;
mod resume;
mod run;
mod state;
mod stopper;
mod trainer_eval;

pub use config::{EarlyStoppingConfig, EvalConfig};
pub use holdout::{HoldoutPlan, HoldoutSplit};
pub use resume::{read_eval_resume, EvalCheckpoint, EvalResume};
pub use run::{evaluate, EvalTotals};
pub use state::{EvalOutcome, Evaluator};
pub use stopper::{Decision, EarlyStopper};
