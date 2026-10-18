# CLI Reference

Complete command-line interface documentation for Oxidizr.

## Overview

Oxidizr uses a subcommand-based CLI with three main commands:

```bash
oxidizr <SUBCOMMAND> [OPTIONS]
```

**Subcommands:**

- `train` - Train a language model
- `pack` - Package a trained model for distribution
- `push` - Push a packaged model to HuggingFace Hub (requires `--features huggingface`)

**Backwards Compatibility:** You can still use `oxidizr -f config.yaml` (without the `train` subcommand) for training. This is equivalent to `oxidizr train -f config.yaml`.

## oxidizr train

Train a language model using a YAML configuration file.

### Basic Usage

```bash
oxidizr train -f <config.yaml> [OPTIONS]

# Or use legacy syntax (backwards compatible):
oxidizr -f <config.yaml> [OPTIONS]
```

When running from source:

```bash
cargo run --release -- train -f <config.yaml> [OPTIONS]
cargo run --release --features cuda -- train -f <config.yaml> [OPTIONS]
```

### Required Arguments

| Argument              | Description                     |
| --------------------- | ------------------------------- |
| `-f, --config <FILE>` | Path to YAML configuration file |

### Data Options

| Argument            | Description                                                                 |
| ------------------- | --------------------------------------------------------------------------- |
| `-d, --data <FILE>` | Path to tokenized data file (.bin). If not specified, generates dummy data. |

### Device Options

| Argument                     | Description                                                           |
| ---------------------------- | --------------------------------------------------------------------- |
| `--target-device <cpu\|gpu>` | Override target device. Default: gpu if CUDA available, otherwise cpu |
| `--gpus <ID>`                | One GPU ID. More than one ID is an error                             |

#### CPU Training

No special flags needed. Just build without CUDA:

```bash
cargo build --release
cargo run --release -- train -f models/nano.yaml --target-device cpu
```

#### GPU Training

Requires CUDA feature:

```bash
cargo build --release --features cuda
cargo run --release --features cuda -- train -f models/nano.yaml
```

#### Multi-GPU Training

Multi-GPU training is not available. `--gpus 0,1` exits with an error. Pass one GPU:

```bash
cargo run --release --features cuda -- train -f models/nano.yaml --gpus 0
```

### Training Overrides

Override config values from command line:

| Argument           | Description                          |
| ------------------ | ------------------------------------ |
| `--seq-len <N>`    | Override sequence length             |
| `--batch-size <N>` | Override batch size                  |
| `--grad-accum <N>` | Override gradient accumulation steps |
| `--max-steps <N>`  | Override maximum training steps      |

Example:

```bash
cargo run --release -- train -f models/nano.yaml --batch-size 4 --seq-len 256 --max-steps 1000
```

### Checkpoint Options

| Argument                | Description                                             |
| ----------------------- | ------------------------------------------------------- |
| `--resume <PATH\|auto>` | Resume from checkpoint. Use `auto` to find latest. Restores weights, optimizer state and data position. See [Checkpoints](training/checkpoints.md). |
| `--max-checkpoints <N>` | Maximum checkpoints to keep. 0 = unlimited. Default: 10 |

Example:

```bash
# Resume from specific checkpoint
cargo run --release -- train -f models/nano.yaml --resume checkpoints/step_1000.safetensors

# Resume from latest checkpoint
cargo run --release -- train -f models/nano.yaml --resume auto
```

### Output Options

| Argument                   | Description                                                      |
| -------------------------- | ---------------------------------------------------------------- |
| `--headless`               | Output JSON metrics only. No TUI progress bar.                   |
| `--run-dir <DIR>`          | Directory for `run.json` and `steps.json`. Default: `$OXIDIZED_RUN_DIR`, else `./runs/<timestamp>`. |
| `--dtype <f32\|f16\|bf16>` | Model precision. Default: f32. BF16 recommended for modern GPUs. |

#### Interactive Mode (Default)

Shows a TUI progress bar with live metrics:

- Current loss
- Training speed (iterations/second)
- ETA to completion
- Memory usage

#### Headless Mode

Use when:

- TUI doesn't render (CI/CD, piped output, remote shells)
- You want to parse metrics programmatically
- Progress bar appears stuck

```bash
cargo run --release -- train -f models/nano.yaml --headless
```

Output format (JSON per line):

```json
{"step": 1, "loss": 11.5161, "grad_norm": 0.2542, "learning_rate": 2.000000e-3, "epoch": 0.01, "it/s": 3.24}
{"step": 21, "loss": 11.4948, "grad_norm": 0.1980, "learning_rate": 2.000000e-3, "epoch": 0.21, "it/s": 3.09}
```

### Performance Options

| Argument         | Description                                                     |
| ---------------- | --------------------------------------------------------------- |
| `--prefetch <N>` | Prefetch N batches in background thread. Default: 0 (disabled). |

Prefetching overlaps CPU data loading with GPU compute:

```bash
cargo run --release --features cuda -- train -f models/nano.yaml --prefetch 2
```

### Full Training Example

```bash
# Full training run with all options
cargo run --release --features cuda -- train \
    -f models/nano.yaml \
    -d data/my_dataset.bin \
    --target-device gpu \
    --batch-size 8 \
    --seq-len 512 \
    --max-steps 10000 \
    --prefetch 2 \
    --dtype bf16 \
    --resume auto
```

## oxidizr pack

Package a trained model for distribution to HuggingFace Hub or local deployment.

### Basic Usage

```bash
oxidizr pack [OPTIONS]

# When running from source:
cargo run --release -- pack [OPTIONS]
```

### Interactive Mode (Default)

Running `oxidizr pack` without arguments launches an interactive TUI:

1. Scans `./checkpoints` directory for `.safetensors` files
2. Displays a list of available checkpoints with metadata
3. Allows you to select which checkpoint to package
4. Packages the model into `hf/<username>/<model>/`

### Non-Interactive Mode

Specify checkpoint via command line flags:

```bash
# Pack latest checkpoint
oxidizr pack --checkpoint latest

# Pack final checkpoint
oxidizr pack --checkpoint final

# Pack specific step
oxidizr pack --checkpoint 10000
```

### Options

| Argument                            | Description                                           | Default           |
| ----------------------------------- | ----------------------------------------------------- | ----------------- |
| `--checkpoint-dir <DIR>`            | Directory to scan for checkpoints                     | `./checkpoints`   |
| `-c, --checkpoint <NAME>`           | Specific checkpoint ("latest", "final", or step #)    | Interactive TUI   |
| `-n, --name <NAME>`                 | Model name for the package                            | Derived from step |
| `-o, --output <DIR>`                | Output directory                                      | `hf/<user>/<model>` |
| `-u, --username <USERNAME>`         | HuggingFace username (overrides `.env` HF_USERNAME)   | From `.env`       |

### Output Structure

Packaged models are created in the following structure:

```
hf/
└── <username>/
    └── <model-name>/
        ├── model.safetensors     # Model weights
        ├── config.json           # Inference configuration
        ├── training_config.json  # Training configuration (optional)
        └── README.md             # Auto-generated model card
```

### Examples

```bash
# Interactive mode
oxidizr pack

# Pack latest checkpoint with custom name
oxidizr pack --checkpoint latest --name my-model

# Pack specific checkpoint to custom location
oxidizr pack \
  --checkpoint-dir ./experiments/run1/checkpoints \
  --checkpoint final \
  --name experiment-final \
  --username my-username \
  --output ./packaged-models
```

### Environment Variables

The pack command uses the following environment variables from `.env`:

| Variable      | Description              | Required |
| ------------- | ------------------------ | -------- |
| `HF_USERNAME` | Your HuggingFace username | No       |

Create a `.env` file in the project root:

```env
HF_USERNAME=your-username
```

You can override this with `--username` flag.

## oxidizr push

Push a packaged model to HuggingFace Hub.

**Note:** This command requires building with the `huggingface` feature:

```bash
cargo build --release --features huggingface
```

Also requires `huggingface-cli` to be installed:

```bash
pip install huggingface_hub
```

### Basic Usage

```bash
oxidizr push [OPTIONS]

# When running from source:
cargo run --release --features huggingface -- push [OPTIONS]
```

### Interactive Mode (Default)

Running `oxidizr push` without arguments:

1. Scans the `hf/` directory for packaged models
2. Displays a list of available models with validation status
3. Allows you to select which model to push
4. Prompts for confirmation before uploading

### Non-Interactive Mode

Specify model via command line flags:

```bash
# Push specific model
oxidizr push --model hf/username/model-name

# Skip confirmation prompt
oxidizr push --model hf/username/model-name --yes
```

### Options

| Argument                  | Description                                          | Default         |
| ------------------------- | ---------------------------------------------------- | --------------- |
| `-m, --model <PATH>`      | Model directory to push (e.g., hf/username/model)    | Interactive TUI |
| `--create-repo`           | Create repository if it doesn't exist                | true            |
| `--private`               | Make repository private                              | false           |
| `-t, --token <TOKEN>`     | HuggingFace token (overrides `.env` HF_TOKEN)        | From `.env`     |
| `-y, --yes`               | Skip confirmation prompt                             | false           |

### Environment Variables

The push command requires a HuggingFace token for authentication:

| Variable   | Description                                           | Required |
| ---------- | ----------------------------------------------------- | -------- |
| `HF_TOKEN` | Your HuggingFace API token (from settings/tokens)     | Yes      |

Create a `.env` file in the project root:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Get your token from [HuggingFace settings](https://huggingface.co/settings/tokens).

You can override this with `--token` flag.

### Examples

```bash
# Interactive mode
oxidizr push

# Push specific model
oxidizr push --model hf/my-username/nano-model

# Push as private repository
oxidizr push --model hf/my-username/nano-model --private

# Non-interactive with token override
oxidizr push \
  --model hf/my-username/nano-model \
  --token hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
  --yes
```

### Validation

Before pushing, the command validates:

- Model directory exists
- `model.safetensors` is present
- `config.json` is present
- Total package size is reasonable

Invalid models will be flagged in the interactive TUI.

## Environment Variables

Global environment variables used across all commands:

| Variable               | Description                                           | Used By       |
| ---------------------- | ----------------------------------------------------- | ------------- |
| `RUST_LOG`             | Set log level (e.g., `RUST_LOG=debug`)                | All           |
| `CUDA_VISIBLE_DEVICES` | Limit visible GPUs (e.g., `CUDA_VISIBLE_DEVICES=0,1`) | train         |
| `HF_USERNAME`          | HuggingFace username for packaging                    | pack          |
| `HF_TOKEN`             | HuggingFace API token for pushing                     | push          |

## Complete Workflow Example

Here's a complete workflow from training to publishing:

```bash
# 1. Train a model
cargo run --release --features cuda -- train \
  -f models/nano.yaml \
  -d data/my_dataset.bin \
  --max-steps 10000 \
  --dtype bf16

# 2. Package the trained model
cargo run --release -- pack \
  --checkpoint final \
  --name my-nano-model \
  --username my-hf-username

# 3. Build with HuggingFace support and push
cargo build --release --features huggingface
cargo run --release --features huggingface -- push \
  --model hf/my-hf-username/my-nano-model
```

## Tips and Troubleshooting

### Training Tips

- Start with small batch size and sequence length, then scale up
- Use `--prefetch 2` to improve data loading performance
- Use `--headless` if the progress bar doesn't render correctly
- Use `--resume auto` to continue from the latest checkpoint

### Packaging Tips

- Run `oxidizr pack` without arguments for interactive selection
- Use meaningful model names with `--name` flag
- Verify package contents in `hf/<username>/<model>/` before pushing

### Publishing Tips

- Ensure `huggingface-cli` is installed: `pip install huggingface_hub`
- Test authentication with: `huggingface-cli whoami`
- Use `--private` flag for private repositories
- Check the model card (README.md) before pushing

### Common Issues

**"huggingface-cli not found"**
- Solution: `pip install huggingface_hub`

**"Authentication failed"**
- Verify token in `.env` file
- Check token has write permissions at [HuggingFace settings](https://huggingface.co/settings/tokens)

**"No checkpoints found"**
- Verify checkpoint directory with `--checkpoint-dir`
- Ensure training completed and saved checkpoints

**Progress bar not showing**
- Use `--headless` flag for JSON output
- Check terminal supports ANSI escape codes
