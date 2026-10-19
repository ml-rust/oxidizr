//! `oxidizr tokenize`: text files to a packed `u32` token file and its sidecar.

mod args;
mod documents;
mod encode;
mod input;
mod output;
mod run;

pub use args::{FormatArg, TokenizeArgs, TxtDocs, DEFAULT_BATCH_BYTES};
pub use run::run_tokenize;
