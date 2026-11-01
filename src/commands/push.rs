//! Push command implementation (feature-gated: huggingface)
//!
//! Pushes a packaged model to HuggingFace Hub using huggingface-cli.

use anyhow::{Context, Result};
use clap::Args;
use dialoguer::{theme::ColorfulTheme, Confirm, Select};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Push command arguments
#[derive(Args, Debug)]
pub struct PushArgs {
    /// Model directory to push (e.g., hf/username/model)
    #[arg(short = 'm', long)]
    pub model: Option<String>,

    /// Create repository if it doesn't exist
    #[arg(long, default_value_t = true)]
    pub create_repo: bool,

    /// Make repository private
    #[arg(long)]
    pub private: bool,

    /// HuggingFace token (overrides .env HF_TOKEN)
    #[arg(short = 't', long)]
    pub token: Option<String>,

    /// Skip confirmation prompt
    #[arg(short = 'y', long)]
    pub yes: bool,
}

/// Information about a packaged model
#[derive(Debug)]
struct PackagedModel {
    path: PathBuf,
    username: String,
    model_name: String,
    has_model: bool,
    has_config: bool,
    #[allow(dead_code)] // Used for display/validation in the future
    has_readme: bool,
}

impl PackagedModel {
    fn is_valid(&self) -> bool {
        self.has_model && self.has_config
    }

    fn display_name(&self) -> String {
        let status = if self.is_valid() { "✓" } else { "⚠" };
        format!("{} {}/{}", status, self.username, self.model_name)
    }

    fn repo_id(&self) -> String {
        format!("{}/{}", self.username, self.model_name)
    }

    fn total_size_mb(&self) -> f64 {
        let mut total = 0u64;
        if let Ok(entries) = fs::read_dir(&self.path) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    total += meta.len();
                }
            }
        }
        total as f64 / (1024.0 * 1024.0)
    }
}

/// Scan hf/ directory for packaged models
fn scan_packaged_models() -> Result<Vec<PackagedModel>> {
    let mut models = Vec::new();
    let hf_dir = Path::new("hf");

    if !hf_dir.exists() {
        return Ok(models);
    }

    // Iterate over username directories
    for user_entry in fs::read_dir(hf_dir)? {
        let user_entry = user_entry?;
        let user_path = user_entry.path();

        if !user_path.is_dir() {
            continue;
        }

        let username = user_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        // Iterate over model directories
        for model_entry in fs::read_dir(&user_path)? {
            let model_entry = model_entry?;
            let model_path = model_entry.path();

            if !model_path.is_dir() {
                continue;
            }

            let model_name = model_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            let has_model = model_path.join("model.safetensors").exists();
            let has_config = model_path.join("config.json").exists();
            let has_readme = model_path.join("README.md").exists();

            models.push(PackagedModel {
                path: model_path,
                username: username.clone(),
                model_name,
                has_model,
                has_config,
                has_readme,
            });
        }
    }

    Ok(models)
}

/// Select a packaged model interactively or by path
fn select_model<'a>(
    models: &'a [PackagedModel],
    selection: Option<&str>,
) -> Result<&'a PackagedModel> {
    match selection {
        Some(path_str) => {
            // Find by path or repo_id
            let path = PathBuf::from(path_str);
            models
                .iter()
                .find(|m| m.path == path || m.repo_id() == path_str)
                .ok_or_else(|| anyhow::anyhow!("Model not found: {}", path_str))
        }
        None => {
            if models.is_empty() {
                return Err(anyhow::anyhow!(
                    "No packaged models found in hf/ directory. Run 'oxidizr pack' first."
                ));
            }

            // Interactive selection
            let items: Vec<String> = models.iter().map(|m| m.display_name()).collect();
            let selection = Select::with_theme(&ColorfulTheme::default())
                .with_prompt("Select a model to push")
                .items(&items)
                .default(0)
                .interact()
                .context("Failed to get user selection")?;

            Ok(&models[selection])
        }
    }
}

/// Check if hf CLI is available
fn check_hf_cli() -> bool {
    // Just check if the command exists and runs (exit code doesn't matter)
    Command::new("hf").arg("--help").output().is_ok()
}

/// Run the push command
pub fn run_push(args: PushArgs) -> Result<()> {
    // Load .env file if it exists
    let _ = dotenvy::dotenv();

    println!("🚀 Oxidizr Model Publisher");
    println!("==========================\n");

    // Check for hf CLI
    if !check_hf_cli() {
        println!("❌ hf CLI not found.\n");
        println!("Install it with:");
        println!("  pip install huggingface_hub\n");
        println!("Then login with:");
        println!("  hf login");
        return Err(anyhow::anyhow!("hf CLI is required for pushing models"));
    }

    // Get HuggingFace token (optional - hf CLI may already be logged in)
    let token = args.token.or_else(|| std::env::var("HF_TOKEN").ok());

    // Scan for packaged models
    let models = scan_packaged_models()?;
    println!("Found {} packaged model(s)\n", models.len());

    // Select model
    let model = select_model(&models, args.model.as_deref())?;

    if !model.is_valid() {
        return Err(anyhow::anyhow!(
            "Model {} is incomplete. Missing: {}{}",
            model.repo_id(),
            if !model.has_model {
                "model.safetensors "
            } else {
                ""
            },
            if !model.has_config { "config.json" } else { "" },
        ));
    }

    // Display upload summary
    println!("📦 Model: {}", model.repo_id());
    println!("📁 Path: {}", model.path.display());
    println!("📊 Size: {:.2} MB", model.total_size_mb());
    println!();

    // List files
    println!("Files to upload:");
    for entry in fs::read_dir(&model.path)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        let size = entry.metadata()?.len() as f64 / (1024.0 * 1024.0);
        println!("  - {} ({:.2} MB)", name, size);
    }
    println!();

    // Confirm upload
    if !args.yes {
        let confirm = Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt(format!(
                "Push to https://huggingface.co/{}?",
                model.repo_id()
            ))
            .default(true)
            .interact()
            .context("Failed to get confirmation")?;

        if !confirm {
            println!("Aborted.");
            return Ok(());
        }
    }

    // Build hf upload command
    let mut cmd = Command::new("hf");
    cmd.arg("upload")
        .arg(model.repo_id())
        .arg(&model.path)
        .arg("--repo-type")
        .arg("model");

    // Add token if provided
    if let Some(ref t) = token {
        cmd.arg("--token").arg(t);
    }

    // Add private flag
    if args.private {
        cmd.arg("--private");
    }

    // hf upload pushes directly to main by default (no --create-pr needed)

    println!("\n⏳ Uploading to HuggingFace Hub...\n");

    // Execute upload
    let status = cmd.status().context("Failed to execute hf CLI")?;

    if status.success() {
        println!("\n✅ Upload complete!");
        println!("🔗 https://huggingface.co/{}", model.repo_id());
    } else {
        return Err(anyhow::anyhow!(
            "Upload failed. Check your token and try again.\n\
             If not logged in, run: hf login"
        ));
    }

    Ok(())
}
