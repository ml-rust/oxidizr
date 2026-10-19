//! Read and write the optimizer state file of a checkpoint set.
//!
//! The file is a SafeTensors file keyed as `Optimizer::state_dict` keys it:
//! `<checkpoint name>.exp_avg`, `<checkpoint name>.exp_avg_sq`,
//! `<checkpoint name>.master`, and `__step`. Every tensor keeps its dtype.

use super::names::NameMaps;
use crate::model::stage_tensor_for_safetensors;
use crate::prelude::{DType, Error, FusedAdamW, Result, Tensor};
use boostr::format::safetensors::save_safetensors;
use boostr::format::SafeTensors;
use boostr::ops::TensorOps;
use boostr::optimizer::Optimizer;
use boostr::runtime::{Runtime, RuntimeClient};
use boostr::{CpuRuntime, TypeConversionOps};
use std::collections::HashMap;
use std::path::Path;

/// `format` value in the file's SafeTensors metadata.
const FORMAT: &str = "oxidizr-optimizer";

fn fail(path: &Path, what: &str, e: impl std::fmt::Display, next: &str) -> Error {
    Error::TrainingError {
        reason: format!(
            "optimizer state file '{}': {what}: {e}. {next}",
            path.display()
        ),
    }
}

fn stage_fail(what: &str, e: impl std::fmt::Display) -> Error {
    Error::TrainingError {
        reason: format!("optimizer state {what}: {e}. Check that host memory is free."),
    }
}

/// Copy the state of `optimizer` to host memory, keyed by checkpoint name.
///
/// Each tensor keeps its own dtype, as the weights do. The copy is required:
/// the next step updates the device state in place.
pub fn stage_optimizer_state<R>(
    optimizer: &FusedAdamW<R>,
    name_maps: &NameMaps,
) -> Result<HashMap<String, Tensor<CpuRuntime>>>
where
    R: Runtime<DType = DType>,
    R::Client: RuntimeClient<R> + TensorOps<R> + TypeConversionOps<R>,
{
    let state = optimizer
        .state_dict(name_maps.names())
        .map_err(|e| stage_fail("export failed", e))?;
    let cpu = boostr::CpuDevice::new();
    let mut staged = HashMap::with_capacity(state.len());
    for (key, tensor) in state {
        let host = stage_tensor_for_safetensors::<R>(&tensor, &cpu)
            .map_err(|e| stage_fail(&format!("copy of '{key}' to host failed"), e))?;
        staged.insert(key, host);
    }
    Ok(staged)
}

/// Write the host optimizer state `staged` to `path`.
///
/// The write is atomic: an error leaves an existing file unchanged.
pub fn write_staged_optimizer_file(
    path: &Path,
    staged: &HashMap<String, Tensor<CpuRuntime>>,
) -> Result<()> {
    let next = "Check that the checkpoint directory is writable and has free space.";
    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), FORMAT.to_string());
    metadata.insert("optimizer".to_string(), "adamw".to_string());
    save_safetensors(path, staged, Some(&metadata)).map_err(|e| fail(path, "write failed", e, next))
}

/// Read every tensor of the optimizer state file at `path` onto `device`.
///
/// Every tensor keeps its stored dtype. F64 entries, such as `__step`, stay F64.
pub fn read_optimizer_file<R>(path: &Path, device: &R::Device) -> Result<HashMap<String, Tensor<R>>>
where
    R: Runtime<DType = DType>,
{
    let next = "Restore the file from the run that wrote it.";
    let mut file = SafeTensors::open(path).map_err(|e| fail(path, "open failed", e, next))?;
    let names: Vec<String> = file.tensor_names().map(str::to_string).collect();
    let mut state = HashMap::with_capacity(names.len());
    for name in names {
        let read_err = |e: Error| fail(path, &format!("read of '{name}' failed"), e, next);
        let (dtype, shape) = {
            let info = file.tensor_info(&name).map_err(read_err)?;
            (info.dtype, info.shape.clone())
        };
        // `SafeTensors::load_tensor` narrows F64 to F32. The step counter must stay F64.
        let tensor = if dtype == DType::F64 {
            let bytes = file.read_tensor_bytes(&name).map_err(read_err)?;
            let values: Vec<f64> = bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b))
                .collect();
            Tensor::<R>::from_slice(&values, &shape, device)
                .map_err(|e| read_err(Error::Numr(e)))?
        } else {
            file.load_tensor::<R>(&name, device).map_err(read_err)?
        };
        state.insert(name, tensor);
    }
    Ok(state)
}

/// Read the optimizer state file at `path` and load it into `optimizer`.
///
/// The load is strict and atomic. An unknown name, a missing key, or a shape
/// or dtype mismatch is an error, and the optimizer stays unchanged.
pub fn load_optimizer_file<R>(
    path: &Path,
    optimizer: &mut FusedAdamW<R>,
    name_maps: &NameMaps,
    device: &R::Device,
) -> Result<()>
where
    R: Runtime<DType = DType>,
{
    let state = read_optimizer_file::<R>(path, device)?;
    optimizer
        .load_state_dict(name_maps.ids(), &state)
        .map_err(|e| {
            fail(
                path,
                "load failed",
                e,
                "The file does not match this model's trainable parameters. Resume with the \
                 config that wrote this checkpoint.",
            )
        })
}
