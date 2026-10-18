//! The one gradient sum of a data-parallel optimizer step.

use crate::prelude::{DType, Error, Result, Tensor};
use boostr::autograd::GradStore;
use boostr::distributed::all_reduce_flat_sum;
use boostr::ops::TensorOps;
use boostr::runtime::{Communicator, Runtime, RuntimeClient};
use boostr::tensor::TensorId;
use std::collections::HashMap;

/// Sum the step's gradients and its logged loss over every rank of `comm`.
///
/// - `ids`: the trainable parameters in model order, the same on every rank.
///   A `TensorId` is local to its process, so the order carries the match.
/// - `params`: the optimizer's tensors. Each gives its parameter's shape.
/// - `local_loss`: this rank's share of the step loss, already weighted by `1 / M`.
///
/// Returns the step loss summed over every rank. Every rank gets the same value.
/// After the call every gradient is f32 and bit-identical on every rank.
/// A parameter no rank produced a gradient for stays without one.
/// The sum is never divided by the world size: [`super::reduce_step_weighting`]
/// already weights every micro-batch by the global count.
///
/// With one rank no collective runs. The gradients keep their bits and
/// `local_loss` returns unchanged.
///
/// # Errors
/// Returns an error when an id has no tensor in `params`, or the collective fails.
/// A collective error leaves peers waiting. The caller must abort the group.
pub fn reduce_step_gradients<R, C>(
    comm: &dyn Communicator,
    client: &C,
    grads: &mut GradStore<R>,
    ids: &[TensorId],
    params: &HashMap<TensorId, Tensor<R>>,
    local_loss: f64,
) -> Result<f64>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
{
    if comm.world_size() <= 1 {
        return Ok(local_loss);
    }
    let mut declared: Vec<(TensorId, &[usize])> = Vec::with_capacity(ids.len());
    for (index, id) in ids.iter().enumerate() {
        let Some(param) = params.get(id) else {
            return Err(Error::TrainingError {
                reason: format!(
                    "trainable parameter {index} ({id:?}) has no tensor in the optimizer map. \
                     The gradient sum needs its shape"
                ),
            });
        };
        declared.push((*id, param.shape()));
    }
    let mut extras = [local_loss];
    all_reduce_flat_sum(comm, client, grads, &declared, &mut extras)?;
    Ok(extras[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::runtime::{LocalCommunicator, NoOpCommunicator};
    use boostr::{CpuClient, CpuDevice, CpuRuntime};

    fn tensor(values: &[f32], device: &CpuDevice) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(values, &[values.len()], device).expect("tensor")
    }

    #[test]
    fn world_one_keeps_the_gradients_and_the_loss() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let id = TensorId::new();
        let grad = tensor(&[0.5, -1.25], &device);
        let mut grads = GradStore::new();
        grads.insert(id, grad.clone());
        let params = HashMap::from([(id, tensor(&[0.0, 0.0], &device))]);
        let loss =
            reduce_step_gradients(&NoOpCommunicator, &client, &mut grads, &[id], &params, 2.5)
                .expect("world 1");
        assert_eq!(loss, 2.5);
        assert!(grads.get(id).expect("grad").shares_storage_with(&grad));
    }

    #[test]
    fn gradients_and_loss_sum_over_ranks() {
        let comms = LocalCommunicator::group(2);
        let results: Vec<(Vec<f32>, f64)> = std::thread::scope(|s| {
            let handles: Vec<_> = comms
                .iter()
                .map(|comm| {
                    s.spawn(move || {
                        let _guard = comm.abort_on_panic();
                        let device = CpuDevice::new();
                        let client = CpuClient::new(device.clone());
                        let id = TensorId::new();
                        let r = comm.rank() as f32;
                        let mut grads = GradStore::new();
                        grads.insert(id, tensor(&[1.0 + r, -r], &device));
                        let params = HashMap::from([(id, tensor(&[0.0, 0.0], &device))]);
                        let local = [0.75, 0.5][comm.rank()];
                        let loss =
                            reduce_step_gradients(comm, &client, &mut grads, &[id], &params, local)
                                .expect("sum");
                        (grads.get(id).expect("grad").to_vec::<f32>(), loss)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank thread"))
                .collect()
        });
        for (grad, loss) in results {
            assert_eq!(grad, vec![3.0, -1.0]);
            assert_eq!(loss, 1.25);
        }
    }

    #[test]
    fn a_parameter_without_a_tensor_is_an_error() {
        let comms = LocalCommunicator::group(2);
        std::thread::scope(|s| {
            for comm in &comms {
                s.spawn(move || {
                    let _guard = comm.abort_on_panic();
                    let client = CpuClient::new(CpuDevice::new());
                    let mut grads = GradStore::<CpuRuntime>::new();
                    let params = HashMap::new();
                    let id = TensorId::new();
                    let result =
                        reduce_step_gradients(comm, &client, &mut grads, &[id], &params, 0.0);
                    assert!(result.is_err());
                });
            }
        });
    }
}
