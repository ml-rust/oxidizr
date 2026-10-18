//! The loss weighting of one optimizer step over every data-parallel rank.

use crate::prelude::{DType, Error, Result, Tensor};
use crate::step_loss::StepWeighting;
use boostr::distributed::all_reduce_tensor;
use boostr::runtime::{Communicator, ReduceOp, Runtime, RuntimeClient};

/// The weighting of one step over every rank of `comm`.
///
/// `local` comes from this rank's micro-batches. With `W` ranks the step has
/// `M = local micro-batches * W` micro-batches, and each micro-batch loss is
/// scaled by `1 / M`. The gradient sum over ranks is never divided by `W`.
///
/// - Unmasked: no collective runs. Each rank adds `mean / M` per micro-batch.
///   The sum over ranks is the mean over all `M` micro-batches.
/// - Masked: one `I64` sum of the scored counts runs before the first forward
///   pass. Every scored token of the step then weighs `1 / G`, with `G` the
///   count over every rank. The sum over ranks is the exact per-token mean.
///
/// Every rank must take the same path, so every rank must mask, or none.
/// With one rank `local` returns unchanged and no collective runs.
///
/// # Errors
/// Returns an error when the collective fails or a count does not fit.
/// A collective error leaves peers waiting. The caller must abort the group.
pub fn reduce_step_weighting<R, C>(
    comm: &dyn Communicator,
    client: &C,
    local: StepWeighting,
) -> Result<StepWeighting>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>,
{
    let world = comm.world_size();
    if world <= 1 {
        return Ok(local);
    }
    let global_scored = match local {
        StepWeighting::MicroBatchMean { .. } => None,
        StepWeighting::ScoredTokens { total_scored, .. } => {
            Some(sum_count::<R, C>(comm, client, total_scored)?)
        }
    };
    local.over_ranks(world, global_scored)
}

/// Sum `count` over every rank of `comm` through a one-element `I64` tensor.
fn sum_count<R, C>(comm: &dyn Communicator, client: &C, count: usize) -> Result<usize>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>,
{
    let local = i64::try_from(count).map_err(|_| Error::TrainingError {
        reason: format!("scored count {count} does not fit in an i64"),
    })?;
    let tensor = Tensor::<R>::from_slice(&[local], &[1], client.device())?;
    // The collective can run on its own stream. It must see the finished upload.
    client.synchronize();
    all_reduce_tensor(comm, &tensor, ReduceOp::Sum)?;
    comm.sync().map_err(|e| Error::DistributedError {
        reason: format!("sync after the scored-count sum failed: {e}"),
    })?;
    let summed = tensor.try_to_vec::<i64>()?;
    let total = summed
        .first()
        .copied()
        .ok_or_else(|| Error::DistributedError {
            reason: "the scored-count sum returned no element".to_string(),
        })?;
    usize::try_from(total).map_err(|_| Error::DistributedError {
        reason: format!(
            "the scored count summed over {} ranks is {total}. Every rank must pass a \
             count of 0 or more",
            comm.world_size()
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::runtime::{LocalCommunicator, NoOpCommunicator};
    use boostr::{CpuClient, CpuDevice, CpuRuntime};

    #[test]
    fn world_one_returns_the_local_weighting() {
        let client = CpuClient::new(CpuDevice::new());
        let local = StepWeighting::ScoredTokens {
            micro_batches: 3,
            total_scored: 7,
        };
        let got = reduce_step_weighting::<CpuRuntime, _>(&NoOpCommunicator, &client, local)
            .expect("world 1");
        assert_eq!(got, local);
    }

    #[test]
    fn masked_counts_sum_and_micro_batches_scale_on_every_rank() {
        let comms = LocalCommunicator::group(2);
        let results: Vec<_> = std::thread::scope(|s| {
            let handles: Vec<_> = comms
                .iter()
                .map(|comm| {
                    s.spawn(move || {
                        let _guard = comm.abort_on_panic();
                        let client = CpuClient::new(CpuDevice::new());
                        let masked = StepWeighting::ScoredTokens {
                            micro_batches: 2,
                            total_scored: [5, 9][comm.rank()],
                        };
                        let unmasked = StepWeighting::MicroBatchMean { micro_batches: 2 };
                        (
                            reduce_step_weighting::<CpuRuntime, _>(comm, &client, masked)
                                .expect("masked"),
                            reduce_step_weighting::<CpuRuntime, _>(comm, &client, unmasked)
                                .expect("unmasked"),
                        )
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank thread"))
                .collect()
        });
        for (masked, unmasked) in results {
            assert_eq!(
                masked,
                StepWeighting::ScoredTokens {
                    micro_batches: 4,
                    total_scored: 14
                }
            );
            assert_eq!(unmasked, StepWeighting::MicroBatchMean { micro_batches: 4 });
        }
    }
}
