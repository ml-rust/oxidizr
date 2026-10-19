//! The fit loop of `LitTrainer` and the sinks that show its progress.

mod checkpoint_flow;
mod fit_loop;
mod json_sink;
mod sink;
mod throughput;
mod tui_sink;

pub use checkpoint_flow::CheckpointQueue;
pub use json_sink::JsonSink;
pub use sink::{CheckpointNote, EvalReport, FitStart, FitSummary, NullSink, StepReport, StepSink};
pub use tui_sink::TuiSink;
