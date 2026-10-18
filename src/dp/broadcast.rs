//! Broadcasts from rank 0: the initial model tensors and the early-stop flag.

use crate::model::LitGPT;
use crate::prelude::{DType, Error, Result, Tensor};
use boostr::distributed::{broadcast_params_ordered, broadcast_tensor};
use boostr::ops::TensorOps;
use boostr::runtime::{Communicator, Runtime, RuntimeClient};
use boostr::tensor::TensorId;
use std::collections::HashMap;

/// Copy rank 0's model tensors into every rank's tensors, in name order.
///
/// Every tensor of `model` travels: the trainable ones from `params`, the
/// frozen ones (a LoRA base) from the model itself. Name order is stable on
/// every rank, so the collectives match. The copy writes through each
/// tensor's storage, so `params` and the model both see rank 0's values.
///
/// Returns the count of tensors sent. With one rank nothing runs and 0 returns.
///
/// # Errors
/// Returns an error when a trainable parameter has no tensor in `params`,
/// a tensor is not contiguous, or a collective fails.
/// A collective error leaves peers waiting. The caller must abort the group.
pub fn broadcast_model_parameters<R, C>(
    comm: &dyn Communicator,
    client: &C,
    model: &LitGPT<R>,
    params: &HashMap<TensorId, Tensor<R>>,
) -> Result<usize>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
{
    if comm.world_size() <= 1 {
        return Ok(0);
    }
    let mut tensors = Vec::new();
    for (name, id, var) in model.named_parameters()? {
        let tensor = match params.get(&id) {
            Some(tensor) => tensor.clone(),
            None if !var.requires_grad() => var.tensor().clone(),
            None => {
                return Err(Error::TrainingError {
                    reason: format!(
                        "trainable parameter '{name}' ({id:?}) has no tensor in the optimizer \
                         map. The initial broadcast needs every model tensor"
                    ),
                });
            }
        };
        tensors.push((name, tensor));
    }
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    broadcast_params_ordered(comm, client, &tensors, 0)?;
    Ok(tensors.len())
}

/// Agree on rank 0's early-stop decision: `stop` on rank 0 reaches every rank.
///
/// Other ranks pass any value. Every rank must call it at the same step
/// boundary. With one rank no collective runs and `stop` returns.
///
/// # Errors
/// Returns an error when the collective fails.
/// A collective error leaves peers waiting. The caller must abort the group.
pub fn broadcast_stop_flag<R, C>(comm: &dyn Communicator, client: &C, stop: bool) -> Result<bool>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>,
{
    if comm.world_size() <= 1 {
        return Ok(stop);
    }
    let flag = Tensor::<R>::from_slice(&[u32::from(stop)], &[1], client.device())?;
    // The collective can run on its own stream. It must see the finished upload.
    client.synchronize();
    broadcast_tensor(comm, &flag, 0)?;
    comm.sync().map_err(|e| Error::DistributedError {
        reason: format!("sync after the early-stop broadcast failed: {e}"),
    })?;
    let value = flag.try_to_vec::<u32>()?;
    Ok(value.first().is_some_and(|&v| v != 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::runtime::{LocalCommunicator, NoOpCommunicator};
    use boostr::{CpuClient, CpuDevice, CpuRuntime};

    #[test]
    fn world_one_keeps_the_local_flag() {
        let client = CpuClient::new(CpuDevice::new());
        for stop in [false, true] {
            let got = broadcast_stop_flag::<CpuRuntime, _>(&NoOpCommunicator, &client, stop)
                .expect("world 1");
            assert_eq!(got, stop);
        }
    }

    #[test]
    fn rank_zero_decides_for_every_rank() {
        let comms = LocalCommunicator::group(3);
        for root_stop in [false, true] {
            let results: Vec<bool> = std::thread::scope(|s| {
                let handles: Vec<_> = comms
                    .iter()
                    .map(|comm| {
                        s.spawn(move || {
                            let _guard = comm.abort_on_panic();
                            let client = CpuClient::new(CpuDevice::new());
                            // Other ranks pass the opposite flag, which must lose.
                            let local = if comm.rank() == 0 {
                                root_stop
                            } else {
                                !root_stop
                            };
                            broadcast_stop_flag::<CpuRuntime, _>(comm, &client, local)
                                .expect("broadcast")
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("rank thread"))
                    .collect()
            });
            assert_eq!(results, vec![root_stop; 3]);
        }
    }
}
