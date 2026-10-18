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
/// At `W > 1` every rank runs one `I64` sum of two counts before the first
/// forward pass, whatever its data. The counts are this rank's scored tokens
/// and 1 when it masks. An unmasked rank sends 0 for both.
///
/// - No rank masks: each rank adds `mean / M` per micro-batch. The sum over
///   ranks is the mean over all `M` micro-batches.
/// - Every rank masks: every scored token of the step weighs `1 / G`, with
///   `G` the scored count over every rank. The sum over ranks is the exact
///   per-token mean.
/// - Some ranks mask and some do not: every rank returns the same error.
///   No gradient work has started, and no rank waits in a later collective.
///
/// With one rank `local` returns unchanged and no collective runs.
///
/// # Errors
/// Returns an error when the collective fails, a count does not fit, or the
/// ranks disagree on the loss mask. A collective error leaves peers waiting.
/// The caller must abort the group.
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
    let (scored, masks) = match local {
        StepWeighting::MicroBatchMean { .. } => (0, 0),
        StepWeighting::ScoredTokens { total_scored, .. } => (total_scored, 1),
    };
    let [global_scored, masked_ranks] = sum_counts::<R, C>(comm, client, [scored, masks])?;
    match masked_ranks {
        0 => local.over_ranks(world, None),
        m if m == world => local.over_ranks(world, Some(global_scored)),
        m => Err(Error::TrainingError {
            reason: format!(
                "{m} of {world} data-parallel ranks carry a loss mask in this step. Every rank \
                 must mask, or none. Build every rank's loader from the same config"
            ),
        }),
    }
}

/// Sum `counts` over every rank of `comm` through one `I64` tensor.
fn sum_counts<R, C>(comm: &dyn Communicator, client: &C, counts: [usize; 2]) -> Result<[usize; 2]>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>,
{
    let mut local = [0i64; 2];
    for (slot, count) in local.iter_mut().zip(counts) {
        *slot = i64::try_from(count).map_err(|_| Error::TrainingError {
            reason: format!("step count {count} does not fit in an i64"),
        })?;
    }
    let tensor = Tensor::<R>::from_slice(local.as_slice(), &[2], client.device())?;
    // The collective can run on its own stream. It must see the finished upload.
    client.synchronize();
    all_reduce_tensor(comm, &tensor, ReduceOp::Sum)?;
    comm.sync().map_err(|e| Error::DistributedError {
        reason: format!("sync after the step-count sum failed: {e}"),
    })?;
    let summed = tensor.try_to_vec::<i64>()?;
    let mut totals = [0usize; 2];
    if summed.len() != totals.len() {
        return Err(Error::DistributedError {
            reason: format!(
                "the step-count sum returned {} elements, expected 2",
                summed.len()
            ),
        });
    }
    for (slot, total) in totals.iter_mut().zip(summed) {
        *slot = usize::try_from(total).map_err(|_| Error::DistributedError {
            reason: format!(
                "a step count summed over {} ranks is {total}. Every rank must pass counts \
                 of 0 or more",
                comm.world_size()
            ),
        })?;
    }
    Ok(totals)
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

    /// Rank 0 masks and rank 1 does not. Both join the same sum, so neither
    /// waits, and both return the same error.
    #[test]
    fn a_masked_and_an_unmasked_rank_both_get_the_mask_error() {
        let comms = LocalCommunicator::group(2);
        let results: Vec<String> = std::thread::scope(|s| {
            let handles: Vec<_> = comms
                .iter()
                .map(|comm| {
                    s.spawn(move || {
                        let _guard = comm.abort_on_panic();
                        let client = CpuClient::new(CpuDevice::new());
                        let local = if comm.rank() == 0 {
                            StepWeighting::ScoredTokens {
                                micro_batches: 1,
                                total_scored: 5,
                            }
                        } else {
                            StepWeighting::MicroBatchMean { micro_batches: 1 }
                        };
                        reduce_step_weighting::<CpuRuntime, _>(comm, &client, local)
                            .expect_err("mixed masks")
                            .to_string()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank thread"))
                .collect()
        });
        for text in &results {
            assert!(text.contains("1 of 2 data-parallel ranks"), "{text}");
            assert!(!text.contains("aborted"), "{text}");
        }
        assert_eq!(results[0], results[1]);
        // The group stays usable: no rank aborted it.
        assert!(comms.iter().all(|c| !c.is_aborted()));
    }
}
