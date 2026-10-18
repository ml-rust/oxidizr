//! GPU id lists: parsing `--gpus`, and the checks a multi-GPU run must pass.

use super::sync::SyncBackend;
use crate::prelude::{Error, Result};

/// What this build supports for more than one GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuildSupport {
    /// Built with `--features cuda`.
    pub cuda: bool,
    /// Built with `--features nccl`, which implies `cuda`.
    pub nccl: bool,
}

impl BuildSupport {
    /// The features of the running build.
    pub fn current() -> Self {
        Self {
            cuda: cfg!(feature = "cuda"),
            nccl: cfg!(feature = "nccl"),
        }
    }
}

/// Parses a comma-separated GPU id list, for example `0,1`.
///
/// # Errors
/// Returns an error naming the first id that is not a non-negative integer.
pub fn parse_gpu_ids(raw: &str) -> Result<Vec<usize>> {
    raw.split(',')
        .map(|id| {
            id.trim()
                .parse::<usize>()
                .map_err(|_| Error::DistributedError {
                    reason: format!(
                        "invalid GPU id '{}' in `--gpus {raw}`. Pass GPU ids as non-negative \
                     integers, for example `--gpus 0,1`.",
                        id.trim()
                    ),
                })
        })
        .collect()
}

/// Returns an error when `gpu_ids` names one GPU twice.
///
/// `input` is the exact user input to quote, for example `--gpus 0,0`.
pub fn reject_duplicate_gpus(input: &str, gpu_ids: &[usize]) -> Result<()> {
    for (i, id) in gpu_ids.iter().enumerate() {
        if gpu_ids[..i].contains(id) {
            return Err(Error::DistributedError {
                reason: format!(
                    "`{input}` names GPU {id} twice. Each rank needs its own GPU. Name each \
                     GPU once."
                ),
            });
        }
    }
    Ok(())
}

/// Checks that a run on `gpu_ids` can start on this build and target.
///
/// One id or none always passes. More than one needs a CUDA target and a
/// build with `cuda` and `nccl`. `uses_cpu` is true when the run trains on the CPU.
///
/// # Errors
/// Returns an error naming `input`, the missing support, and the next action.
pub fn check_gpu_ids(
    input: &str,
    gpu_ids: &[usize],
    uses_cpu: bool,
    build: BuildSupport,
) -> Result<()> {
    reject_duplicate_gpus(input, gpu_ids)?;
    let count = gpu_ids.len();
    if count <= 1 {
        return Ok(());
    }
    let reason = if !build.cuda {
        format!(
            "`{input}` names {count} GPUs, and this build has no CUDA support. Multi-GPU \
             training needs a build with `--features cuda,nccl` and `--target-device gpu`. \
             Rebuild with those features, or pass one GPU id."
        )
    } else if uses_cpu {
        format!(
            "`{input}` names {count} GPUs, and the run trains on the CPU. Multi-GPU training \
             needs `--target-device gpu` and a build with `--features cuda,nccl`. Pass \
             `--target-device gpu`, or remove `--gpus`."
        )
    } else if !build.nccl {
        format!(
            "`{input}` names {count} GPUs, and this build has no NCCL support. Multi-GPU \
             training needs a build with `--features cuda,nccl`. Rebuild with those \
             features, or pass one GPU id, for example `--gpus 0`."
        )
    } else {
        return Ok(());
    };
    Err(Error::DistributedError { reason })
}

/// Checks the sync backend of a run with `world` ranks.
///
/// NCCL is the only multi-rank backend. `explicit` is true when `--sync-backend`
/// named `backend`. The default `single` then runs NCCL for more than one rank.
///
/// # Errors
/// Returns an error at `world > 1` for `cpu`, and for an explicit `single`.
pub fn check_sync_backend(backend: &SyncBackend, explicit: bool, world: usize) -> Result<()> {
    if world <= 1 {
        return Ok(());
    }
    let name = match backend {
        SyncBackend::Nccl => return Ok(()),
        SyncBackend::Single if !explicit => return Ok(()),
        SyncBackend::Single => "single",
        SyncBackend::CpuStaging => "cpu",
    };
    Err(Error::DistributedError {
        reason: format!(
            "sync backend `{name}` cannot run {world} ranks. NCCL is the only multi-rank \
             backend. Pass `--sync-backend nccl`, or remove the flag."
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: BuildSupport = BuildSupport {
        cuda: true,
        nccl: true,
    };

    #[test]
    fn gpu_ids_parse_and_a_bad_id_is_named() {
        assert_eq!(parse_gpu_ids("0, 2").expect("ids"), vec![0, 2]);
        let err = parse_gpu_ids("0,x").expect_err("bad id").to_string();
        assert!(err.contains("'x'") && err.contains("`--gpus 0,x`"), "{err}");
    }

    #[test]
    fn a_duplicate_id_is_an_error_on_every_build() {
        let err = check_gpu_ids("--gpus 1,1", &[1, 1], false, FULL).expect_err("duplicate");
        let text = err.to_string();
        assert!(
            text.contains("`--gpus 1,1`") && text.contains("GPU 1 twice"),
            "{text}"
        );
    }

    #[test]
    fn one_gpu_or_none_passes_everywhere() {
        let none = BuildSupport {
            cuda: false,
            nccl: false,
        };
        for build in [FULL, none] {
            for uses_cpu in [false, true] {
                assert!(check_gpu_ids("--gpus 3", &[3], uses_cpu, build).is_ok());
                assert!(check_gpu_ids("--gpus", &[], uses_cpu, build).is_ok());
            }
        }
    }

    #[test]
    fn each_missing_support_has_its_own_message() {
        let cases = [
            (
                BuildSupport {
                    cuda: false,
                    nccl: false,
                },
                false,
                "no CUDA support",
            ),
            (FULL, true, "trains on the CPU"),
            (
                BuildSupport {
                    cuda: true,
                    nccl: false,
                },
                false,
                "no NCCL support",
            ),
        ];
        for (build, uses_cpu, phrase) in cases {
            let err = check_gpu_ids("--gpus 0,1", &[0, 1], uses_cpu, build).expect_err(phrase);
            let text = err.to_string();
            assert!(text.contains("`--gpus 0,1`"), "{text}");
            assert!(text.contains(phrase), "{text}");
            assert!(text.contains("--features cuda,nccl"), "{text}");
        }
        assert!(check_gpu_ids("--gpus 0,1", &[0, 1], false, FULL).is_ok());
    }

    #[test]
    fn nccl_is_the_only_multi_rank_backend() {
        assert!(check_sync_backend(&SyncBackend::Nccl, true, 2).is_ok());
        assert!(check_sync_backend(&SyncBackend::Single, false, 2).is_ok());
        let err = check_sync_backend(&SyncBackend::Single, true, 2).expect_err("single");
        assert!(
            err.to_string().contains("`single` cannot run 2 ranks"),
            "{err}"
        );
        assert!(check_sync_backend(&SyncBackend::CpuStaging, false, 2).is_err());
        assert!(check_sync_backend(&SyncBackend::CpuStaging, true, 1).is_ok());
    }
}
