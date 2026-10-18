//! Gradient clipping for one optimizer step.
//!
//! The order is the one PyTorch users write by hand: `clip_grad_value_`, then
//! `clip_grad_norm_`. Each element is clamped to `[-max_grad_value,
//! max_grad_value]` first. The global norm clip then scales the clamped
//! gradients to `max_grad_norm`.

use crate::prelude::{compute_grad_norm, DType, Result};
use boostr::autograd::GradStore;
use boostr::ops::TensorOps;
use boostr::optimizer::clip_grad_value;
use boostr::runtime::{Runtime, RuntimeClient};
use boostr::tensor::TensorId;

/// Checks a `trainer.max_grad_value` from the config.
///
/// `None` disables the value clip. A set value must be finite and positive.
pub fn check_max_grad_value(max_grad_value: Option<f64>) -> anyhow::Result<()> {
    match max_grad_value {
        Some(value) if !(value.is_finite() && value > 0.0) => anyhow::bail!(
            "trainer.max_grad_value is {value}: it must be a finite positive number. \
             Set a positive value, or remove the key to disable per-element clipping."
        ),
        _ => Ok(()),
    }
}

/// Clips the gradients of `param_ids` in place and returns a global L2 norm.
///
/// Without a value clip, the returned norm is the raw norm that the norm clip
/// measured. With a value clip, it is the norm after the clamp.
///
/// A clamp can turn NaN or +/-Inf into finite values. The caller skips a step
/// on a non-finite norm, so a value clip measures the raw norm first. A
/// non-finite raw norm returns as is, before any clamp.
pub fn clip_gradients<R, C>(
    client: &C,
    grads: &mut GradStore<R>,
    param_ids: &[TensorId],
    max_grad_value: Option<f64>,
    max_grad_norm: f64,
) -> Result<f64>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + boostr::ScalarOps<R>,
{
    if let Some(max_value) = max_grad_value {
        // `f64::MAX` makes the norm clip a pure measurement: nothing is rescaled.
        let raw_norm = compute_grad_norm(client, grads, param_ids, f64::MAX)?;
        if !raw_norm.is_finite() {
            return Ok(raw_norm);
        }
        clip_grad_value(client, grads, param_ids, max_value)?;
    }
    compute_grad_norm(client, grads, param_ids, max_grad_norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::{CpuClient, CpuDevice, CpuRuntime, Tensor};

    fn store(values: &[&[f32]], device: &CpuDevice) -> (GradStore<CpuRuntime>, Vec<TensorId>) {
        let mut grads = GradStore::new();
        let mut ids = Vec::new();
        for v in values {
            let id = TensorId::new();
            let t = Tensor::<CpuRuntime>::from_slice(v, &[v.len()], device).expect("tensor");
            grads.insert(id, t);
            ids.push(id);
        }
        (grads, ids)
    }

    fn all_values(grads: &GradStore<CpuRuntime>, ids: &[TensorId]) -> Vec<f32> {
        ids.iter()
            .flat_map(|id| grads.get(*id).expect("gradient").to_vec::<f32>())
            .collect()
    }

    #[test]
    fn value_clip_bounds_every_element_of_huge_gradients() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        // Large, but each square still fits in f32, so the raw norm is finite.
        let (mut grads, ids) = store(&[&[1e15, -1e15, 0.25], &[-7e14, 3.0]], &device);

        let norm = clip_gradients(&client, &mut grads, &ids, Some(0.5), f64::MAX).expect("clip");

        let values = all_values(&grads, &ids);
        assert!(values.iter().all(|v| v.abs() <= 0.5), "{values:?}");
        assert_eq!(values[2], 0.25, "an element inside the bound changed");
        // Four elements at 0.5 and one at 0.25: the norm is over clamped values.
        let want = (4.0 * 0.25f64 + 0.0625).sqrt();
        assert!((norm - want).abs() < 1e-6, "norm {norm}, want {want}");
    }

    #[test]
    fn norm_clip_runs_after_the_value_clip() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let (mut grads, ids) = store(&[&[100.0, -100.0, 100.0, -100.0]], &device);

        // Clamp to 3.0 gives norm 6.0. The norm clip then scales it to 1.5.
        let norm = clip_gradients(&client, &mut grads, &ids, Some(3.0), 1.5).expect("clip");

        assert!(
            (norm - 6.0).abs() < 1e-5,
            "the norm clip saw {norm}, not 6.0"
        );
        for v in all_values(&grads, &ids) {
            assert!(
                (v.abs() - 0.75).abs() < 1e-5,
                "element {v} is not 0.75 in size"
            );
        }
    }

    #[test]
    fn disabled_value_clip_leaves_norm_clipping_unchanged() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let (mut grads, ids) = store(&[&[3.0, 4.0]], &device);

        let norm = clip_gradients(&client, &mut grads, &ids, None, f64::MAX).expect("clip");

        assert!((norm - 5.0).abs() < 1e-6);
        assert_eq!(all_values(&grads, &ids), vec![3.0, 4.0]);
    }

    #[test]
    fn a_non_finite_gradient_still_reports_a_non_finite_norm() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let (mut grads, ids) = store(&[&[1.0, bad]], &device);
            let norm = clip_gradients(&client, &mut grads, &ids, Some(0.5), 1.0).expect("clip");
            assert!(
                !norm.is_finite(),
                "{bad} was hidden by the clamp: norm {norm}"
            );
        }
    }

    #[test]
    fn check_max_grad_value_accepts_none_and_positive_values() {
        assert!(check_max_grad_value(None).is_ok());
        assert!(check_max_grad_value(Some(1.0)).is_ok());
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                check_max_grad_value(Some(bad)).is_err(),
                "{bad} was accepted"
            );
        }
    }
}
