//! CLI-flag-over-config resolution for the `train` command.
//!
//! Every flag here follows one rule: it is an `Option`, and it overrides its
//! matching YAML config value only when the caller passes it. Omitting the
//! flag keeps whatever the YAML sets. Each `resolve_*` function returns the
//! chosen value plus whether the CLI supplied it, so the caller logs the
//! source without duplicating the branch.

use crate::config::{DTypePrecision, UnknownDTypePrecision};
use crate::config_paths::{expand_env_vars, PathExpansionError};
use crate::distributed::sync::SyncBackend;

/// Where a resolved value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The `--<flag>` CLI argument was given.
    Cli,
    /// No CLI flag was given; the YAML config's value applies unchanged.
    Config,
}

/// Resolves `model.dtype`: `cli_dtype` overrides it only when `Some`.
pub fn resolve_dtype(
    cli_dtype: Option<&str>,
    config_dtype: DTypePrecision,
) -> Result<(DTypePrecision, Source), UnknownDTypePrecision> {
    match cli_dtype {
        Some(raw) => raw
            .parse::<DTypePrecision>()
            .map(|dtype| (dtype, Source::Cli)),
        None => Ok((config_dtype, Source::Config)),
    }
}

/// Resolves `trainer.distributed.backend`: `cli_backend` overrides it only
/// when `Some`.
pub fn resolve_sync_backend(
    cli_backend: Option<&str>,
    config_backend: SyncBackend,
) -> Result<(SyncBackend, Source), String> {
    match cli_backend {
        Some(raw) => raw
            .parse::<SyncBackend>()
            .map(|backend| (backend, Source::Cli)),
        None => Ok((config_backend, Source::Config)),
    }
}

/// Resolves `trainer.checkpoint_dir`: `cli_checkpoint_dir` overrides it only
/// when `Some`, and — like the YAML value — is run through `${VAR}`
/// expansion, so a CLI override can name the same machine-independent form
/// the config does.
pub fn resolve_checkpoint_dir(
    cli_checkpoint_dir: Option<&str>,
    config_checkpoint_dir: &str,
) -> Result<(String, Source), PathExpansionError> {
    match cli_checkpoint_dir {
        Some(raw) => {
            let expanded = expand_env_vars(raw, "--checkpoint-dir")?;
            Ok((expanded, Source::Cli))
        }
        None => Ok((config_checkpoint_dir.to_string(), Source::Config)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_stays_from_config_when_no_cli_flag_is_given() {
        let (dtype, source) = resolve_dtype(None, DTypePrecision::BF16).expect("no flag, no parse");
        assert_eq!(dtype, DTypePrecision::BF16);
        assert_eq!(source, Source::Config);
    }

    #[test]
    fn dtype_flag_overrides_the_config_value_when_given() {
        let (dtype, source) = resolve_dtype(Some("f16"), DTypePrecision::BF16).expect("f16 parses");
        assert_eq!(dtype, DTypePrecision::F16);
        assert_eq!(source, Source::Cli);
    }

    #[test]
    fn an_unknown_dtype_flag_is_rejected_rather_than_silently_falling_back() {
        let err = resolve_dtype(Some("fp99"), DTypePrecision::BF16).expect_err("unknown dtype");
        assert_eq!(err.0, "fp99");
    }

    #[test]
    fn sync_backend_stays_from_config_when_no_cli_flag_is_given() {
        let (backend, source) =
            resolve_sync_backend(None, SyncBackend::Nccl).expect("no flag, no parse");
        assert_eq!(backend, SyncBackend::Nccl);
        assert_eq!(source, Source::Config);
    }

    #[test]
    fn sync_backend_flag_overrides_the_config_value_when_given() {
        let (backend, source) =
            resolve_sync_backend(Some("nccl"), SyncBackend::Single).expect("nccl parses");
        assert_eq!(backend, SyncBackend::Nccl);
        assert_eq!(source, Source::Cli);
    }

    #[test]
    fn checkpoint_dir_stays_from_config_when_no_cli_flag_is_given() {
        let (dir, source) =
            resolve_checkpoint_dir(None, "./checkpoints").expect("no flag, no expansion");
        assert_eq!(dir, "./checkpoints");
        assert_eq!(source, Source::Config);
    }

    #[test]
    fn checkpoint_dir_flag_overrides_the_config_value_when_given() {
        let (dir, source) = resolve_checkpoint_dir(Some("/tmp/other-checkpoints"), "./checkpoints")
            .expect("no placeholder to expand");
        assert_eq!(dir, "/tmp/other-checkpoints");
        assert_eq!(source, Source::Cli);
    }

    #[test]
    fn checkpoint_dir_flag_expands_env_placeholders_too() {
        unsafe { std::env::set_var("OXIDIZR_TEST_OVERRIDE_DIR", "/tmp/oxidizr-cli-override") };
        let (dir, source) =
            resolve_checkpoint_dir(Some("${OXIDIZR_TEST_OVERRIDE_DIR}/ckpt"), "./checkpoints")
                .expect("variable is set");
        assert_eq!(dir, "/tmp/oxidizr-cli-override/ckpt");
        assert_eq!(source, Source::Cli);
        unsafe { std::env::remove_var("OXIDIZR_TEST_OVERRIDE_DIR") };
    }
}
