//! The loss scaler of one run: scale the loss, unscale the gradients, track overflow.

use super::config::LossScaleConfig;
use crate::prelude::{DType, Result, Tensor};
use boostr::autograd::{var_mul_scalar, GradStore, Var};
use boostr::optimizer::{optimizer_state_dtype, GradScaler, GradScalerState};
use boostr::runtime::{Runtime, RuntimeClient};
use boostr::tensor::TensorId;
use boostr::{ScalarOps, TypeConversionOps};

/// Dynamic loss scaling state of an f16 run.
#[derive(Debug, Clone)]
pub struct LossScaler {
    config: LossScaleConfig,
    scaler: GradScaler,
}

impl LossScaler {
    /// A scaler at `config.initial`.
    ///
    /// # Errors
    /// Returns an error when a field of `config` is out of range.
    pub fn new(config: LossScaleConfig) -> Result<Self> {
        config.check()?;
        let scaler = GradScaler::with_config(config.scaler_config())?;
        Ok(Self { config, scaler })
    }

    /// Continue from a saved `state` under this run's config.
    ///
    /// # Errors
    /// Returns an error when `state.scale` is not finite and positive.
    pub fn restore(&mut self, state: GradScalerState) -> Result<()> {
        self.scaler = GradScaler::from_state(self.config.scaler_config(), state)?;
        Ok(())
    }

    /// The scale the next step multiplies its loss by.
    pub fn scale(&self) -> f64 {
        self.scaler.scale()
    }

    /// Scale and clean-step count, for the checkpoint meta.
    pub fn state(&self) -> GradScalerState {
        self.scaler.state()
    }

    /// Record the outcome of a step. `overflow` is true for a skipped step.
    pub fn record_step(&mut self, overflow: bool) {
        self.scaler.update_scale(overflow);
    }

    /// Divide the gradient of every id in `ids` by the scale, in place.
    ///
    /// Each gradient narrower than f32 is widened to the optimizer state
    /// dtype first. An f16 gradient divided by 65536 in f16 flushes to zero
    /// below 2^-24, which undoes the scaling. The optimizer takes the wide
    /// gradient as is. The work is one cast and one `mul_scalar` per
    /// gradient, with no host read.
    ///
    /// # Errors
    /// Returns an error when a device op fails.
    pub fn unscale<R, C>(
        &self,
        client: &C,
        grads: &mut GradStore<R>,
        ids: &[TensorId],
    ) -> Result<()>
    where
        R: Runtime<DType = DType>,
        C: ScalarOps<R> + TypeConversionOps<R>,
    {
        let inv_scale = 1.0 / self.scale();
        for id in ids {
            let Some(grad) = grads.remove(*id) else {
                continue;
            };
            let wide = optimizer_state_dtype(grad.dtype());
            let grad = if grad.dtype() == wide {
                grad
            } else {
                client.cast(&grad, wide)?
            };
            grads.insert(*id, client.mul_scalar(&grad, inv_scale)?);
        }
        Ok(())
    }
}

/// The two forms of one micro-batch loss: `(logged, backward)`.
///
/// `logged` is `loss / micro_batches`, the value the step reports. Without
/// a scaler, `backward` is the same value as a `Var`, so an unscaled run
/// keeps its bits. With one, `backward` is `loss * scale / micro_batches`.
///
/// # Errors
/// Returns an error when a device op fails.
pub fn weighted_loss<R, C>(
    loss: &Var<R>,
    micro_batches: usize,
    scaler: Option<&LossScaler>,
    client: &C,
) -> Result<(Tensor<R>, Var<R>)>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ScalarOps<R>,
    R::Client: ScalarOps<R>,
{
    let inv = 1.0 / micro_batches as f64;
    match scaler {
        None => {
            let weighted = var_mul_scalar(loss, inv, client)?;
            Ok((weighted.tensor().clone(), weighted))
        }
        Some(scaler) => {
            let logged = client.mul_scalar(loss.tensor(), inv)?;
            let backward = var_mul_scalar(loss, scaler.scale() * inv, client)?;
            Ok((logged, backward))
        }
    }
}

/// Apply a resumed checkpoint's scaler state to the run's scaler.
///
/// - Both present: the run continues from `saved`.
/// - A scaler without `saved` at a step past 0: the scale restarts at
///   `initial`, and a warning says the resume is not exact.
/// - `saved` without a scaler: the run trains in bf16 or f32 now, so the
///   state is dropped with a warning.
///
/// # Errors
/// Returns an error when `saved.scale` is not finite and positive.
pub fn restore_loss_scale(
    scaler: Option<&mut LossScaler>,
    saved: Option<GradScalerState>,
    start_step: usize,
) -> Result<()> {
    match (scaler, saved) {
        (Some(scaler), Some(state)) => scaler.restore(state),
        (Some(scaler), None) if start_step > 0 => {
            eprintln!(
                "WARNING: the checkpoint records no loss scale state. The scale restarts at \
                 {}. Training continues, but it does not match an uninterrupted run.",
                scaler.scale()
            );
            Ok(())
        }
        (None, Some(state)) => {
            eprintln!(
                "WARNING: the checkpoint records loss scale {}, but this run does not train in \
                 f16. The loss scale state is dropped.",
                state.scale
            );
            Ok(())
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::autograd::{backward_wrt, var_add, var_mul, var_sum};
    use boostr::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        (CpuClient::new(device.clone()), device)
    }

    fn scaler(initial: f64) -> LossScaler {
        LossScaler::new(LossScaleConfig {
            initial,
            ..LossScaleConfig::default()
        })
        .expect("scaler")
    }

    /// `loss = sum(w * x + w * b)` over two parameters.
    fn loss(
        w: &Var<CpuRuntime>,
        b: &Var<CpuRuntime>,
        client: &CpuClient,
        device: &CpuDevice,
    ) -> Var<CpuRuntime> {
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.5f32, -0.25], &[2], device).expect("x"),
            false,
        );
        let wx = var_mul(w, &x, client).expect("w * x");
        let wb = var_mul(w, b, client).expect("w * b");
        let both = var_add(&wx, &wb, client).expect("sum");
        var_sum(&both, &[0], false, client).expect("scalar loss")
    }

    #[test]
    fn scaled_then_unscaled_gradient_equals_the_plain_gradient() {
        let (client, device) = setup();
        let w = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.75f32, -2.0], &[2], &device).expect("w"),
            true,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.5f32, 3.0], &[2], &device).expect("b"),
            true,
        );
        let ids = vec![w.id(), b.id()];
        let scaler = scaler(4096.0);

        let l = loss(&w, &b, &client, &device);
        let (_, plain) = weighted_loss(&l, 2, None, &client).expect("plain");
        let want = backward_wrt(&plain, &ids, &client).expect("plain backward");

        let l = loss(&w, &b, &client, &device);
        let (_, scaled) = weighted_loss(&l, 2, Some(&scaler), &client).expect("scaled");
        let mut got = backward_wrt(&scaled, &ids, &client).expect("scaled backward");
        scaler.unscale(&client, &mut got, &ids).expect("unscale");

        for id in &ids {
            let g: Vec<f32> = got.get(*id).expect("grad").to_vec();
            let r: Vec<f32> = want.get(*id).expect("grad").to_vec();
            for (g, r) in g.iter().zip(&r) {
                assert!((g - r).abs() <= 1e-6 * r.abs().max(1.0), "{g} != {r}");
            }
        }
    }

    #[test]
    fn the_logged_loss_is_unscaled() {
        let (client, device) = setup();
        let w = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.75f32, -2.0], &[2], &device).expect("w"),
            true,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.5f32, 3.0], &[2], &device).expect("b"),
            true,
        );
        let l = loss(&w, &b, &client, &device);
        let (plain, _) = weighted_loss(&l, 4, None, &client).expect("plain");
        let (logged, backward) =
            weighted_loss(&l, 4, Some(&scaler(256.0)), &client).expect("scaled");
        let plain: Vec<f32> = plain.to_vec();
        let logged: Vec<f32> = logged.to_vec();
        assert_eq!(plain[0].to_bits(), logged[0].to_bits());
        let backward: Vec<f32> = backward.tensor().to_vec();
        assert_eq!(backward[0], logged[0] * 256.0);
    }

    /// A gradient below f16's smallest subnormal survives: the scaled f16
    /// value holds it, and the unscale runs in f32.
    #[test]
    fn unscale_widens_f16_so_tiny_gradients_survive() {
        let (client, device) = setup();
        let tiny = 1e-8f32;
        let f32_tiny = Tensor::<CpuRuntime>::from_slice(&[tiny], &[1], &device).expect("t");
        let narrow = client.cast(&f32_tiny, DType::F16).expect("cast");
        let back: Vec<f32> = client.cast(&narrow, DType::F32).expect("cast").to_vec();
        assert_eq!(back[0], 0.0, "1e-8 must underflow in f16 without scaling");

        let scaler = scaler(65536.0);
        let scaled =
            Tensor::<CpuRuntime>::from_slice(&[tiny * 65536.0], &[1], &device).expect("scaled");
        let scaled = client.cast(&scaled, DType::F16).expect("cast");
        let id = TensorId::new();
        let mut grads = GradStore::<CpuRuntime>::new();
        grads.insert(id, scaled);
        scaler.unscale(&client, &mut grads, &[id]).expect("unscale");
        let grad = grads.get(id).expect("grad");
        assert_eq!(grad.dtype(), DType::F32);
        let value: Vec<f32> = grad.to_vec();
        assert!((value[0] - tiny).abs() < tiny * 2e-3, "{}", value[0]);
    }

    #[test]
    fn record_step_halves_on_overflow_and_grows_after_the_interval() {
        let mut s = LossScaler::new(LossScaleConfig {
            initial: 64.0,
            growth_interval: 2,
            ..LossScaleConfig::default()
        })
        .expect("scaler");
        s.record_step(true);
        assert_eq!(s.scale(), 32.0);
        s.record_step(false);
        assert_eq!(s.scale(), 32.0);
        s.record_step(false);
        assert_eq!(s.scale(), 64.0);
    }

    #[test]
    fn restore_continues_from_the_saved_state() {
        let mut s = scaler(65536.0);
        let saved = GradScalerState {
            scale: 512.0,
            consecutive_ok: 7,
        };
        restore_loss_scale(Some(&mut s), Some(saved), 3).expect("restore");
        assert_eq!(s.state(), saved);
        let bad = GradScalerState {
            scale: f64::NAN,
            consecutive_ok: 0,
        };
        assert!(restore_loss_scale(Some(&mut s), Some(bad), 3).is_err());
        restore_loss_scale(None, Some(saved), 3).expect("dropped with a warning");
        restore_loss_scale(Some(&mut s), None, 0).expect("fresh run");
    }
}
