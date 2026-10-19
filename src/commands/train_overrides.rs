//! Applies the `train` CLI flags to the loaded config, and prints each change.

use crate::commands::overrides::{
    resolve_checkpoint_dir, resolve_dtype, resolve_sync_backend, Source,
};
use crate::commands::TrainArgs;
use crate::config::LitConfig;
use anyhow::Result;

/// Applies every override flag of `args` to `config`.
///
/// Each flag overrides its config value only when given. Unless `headless`,
/// a line names each change. `--gpus` is applied by the run plan, which also
/// checks the GPU ids.
///
/// Returns where the sync backend came from, `Source::Cli` for `--sync-backend`.
///
/// # Errors
/// Returns an error for an unknown `--sync-backend` or a bad `--checkpoint-dir`.
pub fn apply_cli_overrides(args: &TrainArgs, config: &mut LitConfig) -> Result<Source> {
    let say = !args.headless;
    let trainer = &mut config.trainer;
    let overrides = [
        ("seq_length", args.seq_length, &mut config.model.max_seq_len),
        ("batch_size", args.batch_size, &mut trainer.batch_size),
        (
            "gradient_accumulation",
            args.gradient_accumulation,
            &mut trainer.gradient_accumulation,
        ),
        ("max_steps", args.max_steps, &mut trainer.max_steps),
        (
            "max_checkpoints",
            args.max_checkpoints,
            &mut trainer.max_checkpoints,
        ),
    ];
    for (name, value, slot) in overrides {
        if let Some(value) = value {
            if say {
                println!("CLI override: {name} = {value} (was {slot})");
            }
            *slot = value;
        }
    }
    if args.async_checkpoint {
        if say {
            println!(
                "CLI override: async_checkpoint = true (was {})",
                config.trainer.async_checkpoint
            );
        }
        config.trainer.async_checkpoint = true;
    }

    let current_backend = config.trainer.distributed.backend.clone();
    let backend_source = match resolve_sync_backend(args.sync_backend.as_deref(), current_backend) {
        Ok((backend, source)) => {
            config.trainer.distributed.backend = backend;
            source
        }
        Err(e) => anyhow::bail!("invalid --sync-backend: {e}"),
    };

    // `--dtype` wins only when given. model.dtype from the YAML applies otherwise.
    match resolve_dtype(args.dtype.as_deref(), config.model.dtype) {
        Ok((dtype, source)) => {
            if say {
                match source {
                    Source::Cli => println!(
                        "dtype = {} (source: --dtype CLI flag, config had {})",
                        dtype, config.model.dtype
                    ),
                    Source::Config => println!("dtype = {} (source: model.dtype in config)", dtype),
                }
            }
            config.model.dtype = dtype;
        }
        Err(e) => {
            eprintln!(
                "Warning: {e}, using config default ({})",
                config.model.dtype
            );
        }
    }

    // `--checkpoint-dir` wins only when given, and runs through `${VAR}`
    // expansion. The config's already-expanded value applies otherwise.
    match resolve_checkpoint_dir(
        args.checkpoint_dir.as_deref(),
        &config.trainer.checkpoint_dir,
    ) {
        Ok((checkpoint_dir, source)) => {
            if say {
                match source {
                    Source::Cli => println!(
                        "checkpoint_dir = {} (source: --checkpoint-dir CLI flag, config had {})",
                        checkpoint_dir, config.trainer.checkpoint_dir
                    ),
                    Source::Config => println!(
                        "checkpoint_dir = {} (source: trainer.checkpoint_dir in config)",
                        checkpoint_dir
                    ),
                }
            }
            config.trainer.checkpoint_dir = checkpoint_dir;
        }
        Err(e) => anyhow::bail!("invalid --checkpoint-dir: {e}"),
    }
    Ok(backend_source)
}
