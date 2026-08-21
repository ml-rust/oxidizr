//! Policy for reconciling a checkpoint's embedding/head row count with
//! `ModelConfig::vocab_size`.
//!
//! The math (mean-init row growth) lives in boostr's
//! `nn::resize_rows_mean_init`, which stays architecture-agnostic. This
//! module owns only the decision of when growth is allowed, keyed off
//! `ModelConfig::grow_vocab`.

use crate::prelude::{DType, Error, Result, Tensor};
use boostr::{ReduceOps, Runtime, ShapeOps};

/// Reconcile `weight`'s row count with `target_rows`.
///
/// - Row count already matches: returned unchanged.
/// - Row count is smaller and `grow_vocab` is set: grown via mean-init.
///   Mean-init, not random: a freshly initialized row sits outside the
///   distribution the pretrained rows were trained under, and at low
///   precision (BF16/FP16) that mismatch produces outsized gradients on the
///   very first step and can NaN the run before the new rows ever see real
///   data. The column-wise mean of the existing rows is already inside that
///   distribution, so training starts from a stable point.
/// - Row count is smaller and `grow_vocab` is not set: hard error naming
///   `what`, both row counts, and the opt-in flag.
/// - Row count is *larger*: returned unchanged. Checkpoints routinely pad the
///   embedding table above the declared `vocab_size` (a multiple of 64 or 128
///   for kernel alignment), so this is a normal load, not a mismatch.
///
/// The hard error on the *smaller* case is the point of this function.
/// Growing without being asked would hide a wrong-config/wrong-checkpoint
/// pairing: the model would train and look fine while the rows it was told
/// to use do not exist in the checkpoint.
pub fn fit_vocab_rows<R, C>(
    client: &C,
    weight: &Tensor<R>,
    target_rows: usize,
    grow_vocab: bool,
    what: &str,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ReduceOps<R> + ShapeOps<R>,
{
    let rows = weight.shape().first().copied().unwrap_or(0);

    // `>=` and not `==`: a checkpoint whose table is padded above the
    // declared vocab_size is a normal, widely used layout, and rejecting it
    // would break loads that worked before this check existed.
    if rows >= target_rows {
        return Ok(weight.clone());
    }

    if grow_vocab {
        return boostr::nn::resize_rows_mean_init(client, weight, target_rows);
    }

    Err(Error::ModelError {
        reason: format!(
            "{what}: checkpoint has {rows} rows but config.vocab_size is {target_rows}; \
             the checkpoint cannot supply the missing {} rows. Set grow_vocab: true \
             to mean-init them, or correct the config to match the checkpoint",
            target_rows - rows
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use boostr::{CpuClient, CpuDevice, CpuRuntime};

    fn weight(device: &CpuDevice) -> Tensor<CpuRuntime> {
        #[rustfmt::skip]
        let w = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 3.0, 4.0, 5.0],
            &[2, 3],
            device,
        );
        w
    }

    #[test]
    fn passes_through_when_rows_already_match() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let out = fit_vocab_rows(&client, &weight(&device), 2, false, "w").unwrap();
        assert_eq!(out.shape(), &[2, 3]);
    }

    #[test]
    fn grows_when_flag_is_set() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let out = fit_vocab_rows(&client, &weight(&device), 4, true, "w").unwrap();
        assert_eq!(out.shape(), &[4, 3]);
        let grown = out.to_vec::<f32>();
        // Mean of [1,2,3] and [3,4,5] column-wise is [2,3,4], appended twice.
        assert_eq!(&grown[6..9], &[2.0, 3.0, 4.0]);
        assert_eq!(&grown[9..12], &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn errors_when_flag_is_unset() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        assert!(fit_vocab_rows(&client, &weight(&device), 4, false, "w").is_err());
    }

    /// A table padded above `target_rows` is a normal checkpoint layout, not
    /// a mismatch — it must load untouched whether or not growth is enabled.
    #[test]
    fn padded_checkpoint_passes_through_unchanged() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        for grow in [false, true] {
            let out = fit_vocab_rows(&client, &weight(&device), 1, grow, "w").unwrap();
            assert_eq!(out.shape(), &[2, 3], "grow_vocab={grow}");
        }
    }
}
