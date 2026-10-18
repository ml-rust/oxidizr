//! The fit loop of `LitTrainer` and the sinks that show its progress.

mod fit_loop;
mod json_sink;
mod sink;
mod throughput;
mod tui_sink;

pub use json_sink::JsonSink;
pub use sink::{CheckpointNote, FitStart, FitSummary, StepReport, StepSink};
pub use tui_sink::TuiSink;
