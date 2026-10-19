//! Weighted mixtures of data sources: per-epoch quotas, the epoch schedule,
//! the per-source draws, the loader builder, and the checkpoint record.

mod draw;
mod loader;
mod quota;
mod record;
mod schedule;
mod state;

pub(super) use quota::weight_error;
pub use record::{check_mixture_resume, MixtureRecord, MixtureSource};
pub(super) use state::Mixture;
