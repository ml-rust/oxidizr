use anyhow::Result;
use clap::{Parser, Subcommand};

use oxidizr::commands::{self, MergeArgs, PackArgs, TrainArgs};

#[cfg(feature = "huggingface")]
use oxidizr::commands::PushArgs;

/// Oxidizr: Rust-based ML training framework for GPT-style language models
#[derive(Parser, Debug)]
#[command(name = "oxidizr")]
#[command(about = "A Rust-based ML training framework", long_about = None)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    // Legacy: Allow running train without subcommand for backwards compatibility
    /// Path to the configuration YAML file (shortcut for 'oxidizr train -f')
    #[arg(short = 'f', long = "config", global = true)]
    config_file: Option<String>,

    /// Path to training data file (shortcut for 'oxidizr train -d')
    #[arg(short = 'd', long = "data", global = true)]
    data_file: Option<String>,

    /// Override target device (shortcut for 'oxidizr train --target-device')
    #[arg(long = "target-device", global = true)]
    target_device: Option<String>,

    /// Override sequence length (shortcut for 'oxidizr train --seq-len')
    #[arg(long = "seq-len", global = true)]
    seq_length: Option<usize>,

    /// Override batch size (shortcut for 'oxidizr train --batch-size')
    #[arg(long = "batch-size", global = true)]
    batch_size: Option<usize>,

    /// Override gradient accumulation (shortcut for 'oxidizr train --grad-accum')
    #[arg(long = "grad-accum", global = true)]
    gradient_accumulation: Option<usize>,

    /// Headless mode (shortcut for 'oxidizr train --headless')
    #[arg(long, global = true)]
    headless: bool,

    /// Override max training steps (shortcut for 'oxidizr train --max-steps')
    #[arg(long = "max-steps", global = true)]
    max_steps: Option<usize>,

    /// Resume from checkpoint (shortcut for 'oxidizr train --resume')
    #[arg(long = "resume", global = true)]
    resume: Option<String>,

    /// GPU IDs for multi-GPU training (shortcut for 'oxidizr train --gpus')
    #[arg(long = "gpus", global = true)]
    gpus: Option<String>,

    /// Gradient sync backend (shortcut for 'oxidizr train --sync-backend')
    #[arg(long = "sync-backend", global = true)]
    sync_backend: Option<String>,

    /// Prefetch batches (shortcut for 'oxidizr train --prefetch')
    #[arg(long = "prefetch", global = true)]
    prefetch: Option<usize>,

    /// Max checkpoints to keep (shortcut for 'oxidizr train --max-checkpoints')
    #[arg(long = "max-checkpoints", global = true)]
    max_checkpoints: Option<usize>,

    /// Model precision (shortcut for 'oxidizr train --dtype')
    #[arg(long = "dtype", global = true)]
    dtype: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Train a model (default command)
    Train(TrainArgs),

    /// Package a trained model for distribution
    Pack(PackArgs),

    /// Merge LoRA adapters into base weights, writing a plain checkpoint
    Merge(MergeArgs),

    /// Push a packaged model to HuggingFace Hub
    #[cfg(feature = "huggingface")]
    Push(PushArgs),
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    // Set custom panic hook to suppress backtraces for known recoverable errors (like OOM)
    // These panics are caught by catch_unwind in trainer.rs and converted to user-friendly errors
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        let panic_msg = panic_info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| panic_info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_default();

        // Skip backtrace for OOM errors - they're handled by catch_unwind
        if panic_msg.contains("can't allocate buffer") || panic_msg.contains("out of memory") {
            // Silent - the error will be caught and reported nicely
            return;
        }

        // For other panics, use the default hook
        default_hook(panic_info);
    }));

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Train(args)) => commands::run_train(args),
        Some(Commands::Pack(args)) => commands::run_pack(args),
        Some(Commands::Merge(args)) => commands::run_merge(args),
        #[cfg(feature = "huggingface")]
        Some(Commands::Push(args)) => commands::run_push(args),
        None => {
            // Legacy mode: if -f is provided without subcommand, run train
            if cli.config_file.is_some() {
                let args = TrainArgs {
                    config_file: cli
                        .config_file
                        .unwrap_or_else(|| "models/nano.yaml".to_string()),
                    data_file: cli.data_file,
                    target_device: cli.target_device,
                    seq_length: cli.seq_length,
                    batch_size: cli.batch_size,
                    gradient_accumulation: cli.gradient_accumulation,
                    headless: cli.headless,
                    max_steps: cli.max_steps,
                    resume: cli.resume,
                    gpus: cli.gpus,
                    sync_backend: cli.sync_backend.unwrap_or_else(|| "cpu".to_string()),
                    prefetch: cli.prefetch.unwrap_or(0),
                    max_checkpoints: cli.max_checkpoints,
                    dtype: cli.dtype.unwrap_or_else(|| "f32".to_string()),
                };
                commands::run_train(args)
            } else {
                // No command and no -f flag - show help
                println!("Oxidizr: A Rust-based ML training framework\n");
                println!("Usage:");
                println!("  oxidizr train -f <config.yaml>     Train a model");
                println!("  oxidizr pack                       Package a trained model");
                println!(
                "  oxidizr merge -f <config.yaml> -c <lora.safetensors> -o <merged.safetensors>"
            );
                println!(
                    "                                     Merge LoRA adapters into base weights"
                );
                #[cfg(feature = "huggingface")]
                println!("  oxidizr push                       Push model to HuggingFace");
                println!("\nLegacy usage (backwards compatible):");
                println!("  oxidizr -f <config.yaml>           Same as 'oxidizr train -f'");
                println!("\nFor more options, use: oxidizr --help");
                Ok(())
            }
        }
    }
}
