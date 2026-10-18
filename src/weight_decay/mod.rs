//! Which parameters AdamW weight decay applies to, and the parameter group that carries it.
//!
//! The rule is run config, not optimizer state: the optimizer state file holds
//! only moments, master weights and the step count. The checkpoint meta
//! records the rule, and a resume inherits it, so a resumed run stays exact.

mod plan;
mod resolve;
mod rule;

pub use plan::{DecayPlan, TensorInfo};
pub use resolve::{inherited_weight_decay, resolve_weight_decay};
pub use rule::{WeightDecayRule, WeightDecaySpec};
