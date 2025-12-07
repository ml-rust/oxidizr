# Oxidizr

A Rust-based LLM training framework built on [Candle](https://github.com/huggingface/candle). Oxidizr is a flexible trainer - bring your own config and dataset, and start training.

[Full Documentation](docs/README.md) | [Architecture Guide](docs/architecture/overview.md) | [CLI Reference](docs/cli.md)

## Installation

**Recommended: Install from Git** (supports CUDA 12.x and 13.x):

```bash
# CPU-only
cargo install --git https://github.com/ml-rust/oxidizr

# With CUDA support
cargo install --git https://github.com/ml-rust/oxidizr --features cuda

# With CUDA + HuggingFace model publishing
cargo install --git https://github.com/ml-rust/oxidizr --features cuda,huggingface
```

**From crates.io** (CUDA 12.x only):

```bash
cargo install oxidizr
cargo install oxidizr --features cuda
cargo install oxidizr --features cuda,huggingface
```

> **Note:** The crates.io version only supports CUDA 12.x. For CUDA 13.x support, install from Git.

## Quick Start

```bash
# Clone and build
git clone https://github.com/ml-rust/oxidizr
cd oxidizr
cargo build --release

# Download nano-start dataset
pip install huggingface_hub
hf download fs90/nano-start-data-bin --local-dir data/nano-start/tokenized --repo-type dataset

# Train!
cargo run --release -- train -f models/nano-start.yaml -d data/nano-start/tokenized/combined.bin
```

That's it! Training works on CPU out of the box - no GPU required.

**For faster training with GPU:**

```bash
cargo build --release --features cuda
cargo run --release --features cuda -- train -f models/nano-start.yaml -d data/nano-start/tokenized/combined.bin
```

GPU training is significantly faster but completely optional. CPU training is fully functional, just slower.

**After training, package and share your model:**

```bash
# Package the trained model
cargo run --release -- pack

# Push to HuggingFace (requires --features huggingface)
cargo run --release --features huggingface -- push
```

## What is Oxidizr?

Oxidizr is a **production-grade LLM trainer** written in Rust. You provide:

1. **A config file** - Model architecture, hyperparameters, training settings
2. **A dataset** - Tokenized data (binary format or in-memory)

Oxidizr handles the training loop, optimization, checkpointing, and logging.

### Why Oxidizr?

- **Production-grade** - Train real models, not just toys. Modern architectures (Mamba, MLA, MoE) ready to use.
- **No GPU required** - Full training support on CPU. Great for learning, prototyping, or when GPU isn't available.
- **Researchers & Students** - Transparent codebase, easy to understand and modify. Perfect for experiments.
- **Fast iteration** - Rust performance without Python overhead. Quick feedback loops.
- **Portable** - Single binary, no complex dependencies. Works anywhere Rust compiles.

### Sample Configs Included

We include several example configs in `models/` to help you get started:

- `nano-start.yaml` - Educational config for beginners (cl100k_base vocab)
- `nano.yaml` - Hybrid Mamba2 + MLA + MoE (~60M params)
- `nano_mamba3.yaml` - Pure Mamba3 architecture
- `nano_mamba3_hybrid.yaml` - Hybrid Mamba3 + MLA + MoE

These are **educational examples** showing you how to configure oxidizr. Feel free to create your own configs for your specific use case.

## Bring Your Own Config and Data

### Creating Your Config

Create a YAML file with your model architecture and training settings:

```yaml
# my_model.yaml
model:
  hidden_size: 512
  num_layers: 8
  num_heads: 8
  kv_heads: 4
  vocab_size: 128354 # Llama 3 + splintr agent tokens
  max_seq_len: 512
  rope_theta: 10000.0
  intermediate_size: 2048

trainer:
  learning_rate: 0.0003
  batch_size: 2
  max_steps: 5000
  num_epochs: 2
  gradient_accumulation: 1
  checkpoint_dir: "./checkpoints"
  log_interval: 10
  save_interval: 500
```

Run it:

```bash
cargo run --release --features cuda -- train -f my_model.yaml
```

### Preparing Your Data

Oxidizr accepts tokenized data in binary format (u32 tokens):

**Option 1: Use the educational dataset**

The `data/nano-start/` directory contains a curated educational dataset designed to help you understand LLM training fundamentals. See the `data/` directory for details.

**Option 2: Bring your own tokenized data**

Create a binary file containing raw u32 tokens:

```python
# Using splintr tokenizer (recommended)
from splintr import Tokenizer

tokenizer = Tokenizer("llama3")
tokens = tokenizer.encode("Your training text here...")

# Save as binary u32 array
import numpy as np
np.array(tokens, dtype=np.uint32).tofile("data/my_dataset.bin")
```

Then point your config to the data file, or load it programmatically in your training script.

**Option 3: Generate dummy data for testing**

For quick testing, oxidizr can generate random tokens:

```rust
use oxidizr::data::{LitDataLoader, create_dummy_data};

let tokens = create_dummy_data(128354, 100_000);  // vocab_size, num_tokens
let data_loader = LitDataLoader::new(tokens, batch_size, seq_len, device);
```

## Supported Architectures

Oxidizr supports multiple architectures:

### Base Architectures

- **GPT/Llama-style Transformer** - RoPE, RMSNorm, Grouped Query Attention (GQA), SwiGLU
- **Mamba1** - State Space Model with selective mechanism for efficient long-range context
- **Mamba2** - State Space Duality (SSD) algorithm, faster than Mamba1
- **Mamba3** - Latest Mamba with trapezoidal discretization, complex-valued RoPE, and MIMO

### Advanced Components

- **MLA (Multi-Head Latent Attention)** - Compressed KV cache for memory efficiency
- **MoE (Mixture of Experts)** - Fine-grained expert routing with load balancing

### Hybrid Architectures

You can mix and match components. For example, the `nano_mamba2.yaml` config uses:

- 6 Mamba2 layers for efficient sequential processing
- 2 MLA + MoE layers for cross-sequence attention

Configure hybrid models by specifying which layers use which architecture in your YAML.

## CLI Reference

Oxidizr uses a subcommand-based CLI:

```bash
oxidizr <SUBCOMMAND> [OPTIONS]
```

### Subcommands

- `train` - Train a model (default if -f flag is used)
- `pack` - Package a trained model for distribution
- `push` - Push a packaged model to HuggingFace Hub (requires `--features huggingface`)

### Training

```bash
oxidizr train -f <config.yaml> [OPTIONS]

# Or use the legacy shortcut (backwards compatible):
oxidizr -f <config.yaml> [OPTIONS]

Options:
  -f, --config <FILE>           Path to YAML configuration file (required)
  -d, --data <FILE>             Path to tokenized data file (.bin)
  --target-device <gpu|cpu>     Override target device (default: gpu if available)
  --seq-len <N>                 Override sequence length from config
  --batch-size <N>              Override batch size from config
  --grad-accum <N>              Override gradient accumulation from config
  --max-steps <N>               Override max training steps from config
  --gpus <IDS>                  Comma-separated GPU IDs for multi-GPU (e.g., 0,1,2,3)
  --sync-backend <cpu|nccl>     Gradient sync backend for multi-GPU (default: cpu)
  --prefetch <N>                Prefetch N batches in background (default: 0)
  --resume <PATH|auto>          Resume from checkpoint (.safetensors) or "auto" for latest
  --headless                    Output JSON metrics only (for non-interactive terminals)
  --dtype <f32|f16|bf16>        Model precision (default: f32)
  --max-checkpoints <N>         Maximum checkpoints to keep (default: 10)
  -h, --help                    Print help information
```

### Examples

```bash
# Basic training with default settings
cargo run --release --features cuda -- train -f models/nano.yaml

# Legacy syntax (backwards compatible)
cargo run --release --features cuda -- -f models/nano.yaml

# Force CPU execution
cargo run --release -- train -f models/nano.yaml --target-device cpu

# Override batch size and sequence length
cargo run --release --features cuda -- train -f models/nano.yaml --batch-size 4 --seq-len 256

# Multi-GPU training (2 GPUs)
cargo run --release --features cuda -- train -f models/nano.yaml --gpus 0,1 --sync-backend cpu

# Custom config file
cargo run --release --features cuda -- train -f experiments/my_config.yaml
```

### Output Modes

**Interactive mode (default):**

- Shows a TUI progress bar with live loss, speed, and ETA
- Best for interactive terminal sessions

**Headless mode (`--headless`):**

- Outputs JSON metrics to stdout
- Use when: TUI doesn't render (CI/CD, piped output, non-interactive shells)
- Use when: You want to parse training metrics programmatically

```bash
# If progress bar doesn't appear, use headless mode
cargo run --release -- train -f models/nano.yaml --headless
```

### CPU vs GPU Training

```bash
# CPU training (no CUDA required)
cargo build --release
cargo run --release -- train -f models/nano.yaml --target-device cpu

# GPU training (faster, requires CUDA)
cargo build --release --features cuda
cargo run --release --features cuda -- train -f models/nano.yaml --target-device gpu
```

CPU training is fully functional - just slower. Great for:

- Learning and experimentation
- Systems without GPU
- Debugging and development

## Model Distribution

After training, you can package and share your models:

### Packaging Models

```bash
# Interactive mode - select checkpoint from TUI
oxidizr pack

# Non-interactive - specify checkpoint
oxidizr pack --checkpoint latest
oxidizr pack --checkpoint final
oxidizr pack --checkpoint 10000  # specific step

# Custom options
oxidizr pack \
  --checkpoint-dir ./checkpoints \
  --checkpoint final \
  --name my-model \
  --username my-hf-username
```

This creates a packaged model in `hf/<username>/<model>/` with:

- `model.safetensors` - Model weights
- `config.json` - Inference configuration
- `README.md` - Auto-generated model card

### Publishing to HuggingFace

Requires the `huggingface` feature flag and `huggingface-cli`:

```bash
# Build with HuggingFace support
cargo build --release --features huggingface

# Install HuggingFace CLI
pip install huggingface_hub

# Interactive mode - select model from list
oxidizr push

# Non-interactive - specify model
oxidizr push --model hf/username/model-name

# Create private repository
oxidizr push --model hf/username/model-name --private
```

**Configuration**: Create a `.env` file (see `.env.example`):

```env
HF_USERNAME=your-username
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Get your token from [HuggingFace settings](https://huggingface.co/settings/tokens).

See [hf/README.md](hf/README.md) for detailed documentation.

## Multi-GPU Training

Oxidizr supports data-parallel training across multiple GPUs:

```bash
# Train on GPUs 0, 1, 2, 3 with CPU backend
cargo run --release --features cuda -- train -f models/nano.yaml --gpus 0,1,2,3 --sync-backend cpu

# Train with NCCL backend (faster for 4+ GPUs, requires nccl feature)
cargo run --release --features cuda,nccl -- train -f models/nano.yaml --gpus 0,1 --sync-backend nccl
```

**How it works:**

1. Dataset is sharded across GPUs
2. Each GPU runs forward/backward pass on its shard
3. Gradients are synchronized (all-reduced) across all GPUs
4. Averaged gradients are applied by the optimizer

**Effective batch size** = `batch_size × gradient_accumulation × num_gpus`

## Configuration Guide

### Basic Config Structure

```yaml
model:
  # Architecture parameters
  hidden_size: 512
  num_layers: 8
  num_heads: 8
  kv_heads: 4 # For GQA (fewer KV heads than Q heads)
  vocab_size: 128354 # Llama 3 + splintr agent tokens
  max_seq_len: 512
  rope_theta: 10000.0
  intermediate_size: 2048

trainer:
  # Training hyperparameters
  learning_rate: 0.0003
  batch_size: 2
  max_steps: 5000
  num_epochs: 2
  gradient_accumulation: 1
  checkpoint_dir: "./checkpoints"
  log_interval: 10
  save_interval: 500
  load_balance_alpha: 0.0 # MoE load balancing (0.0 = disabled)
```

### Enabling Mamba2

Add these fields to use Mamba2 instead of standard attention:

```yaml
model:
  # ... other fields ...

  mamba2_num_heads: 48
  mamba2_head_dim: 16
  mamba2_state_size: 64
  mamba2_chunk_size: 64
  mamba2_n_groups: 1
  mamba2_conv_kernel: 4
  mamba2_expand: 2

  # CONSTRAINT: hidden_size * mamba2_expand == mamba2_num_heads * mamba2_head_dim
  # Example: 384 * 2 = 768 == 48 * 16 ✓
```

### Enabling Mamba3

Mamba3 extends Mamba2 with three innovations:

```yaml
model:
  # ... Mamba2 base params (same as above) ...

  # Mamba3 features
  mamba3_enabled: true
  mamba3_complex_rope: true # Complex-valued RoPE for state tracking
  mamba3_mimo_rank: 0 # 0 = SISO, 4 = MIMO mode
  mamba3_use_conv: false # false = trapezoidal discretization
```

### Enabling MLA (Multi-Head Latent Attention)

For compressed KV cache and memory efficiency:

```yaml
model:
  # ... other fields ...

  kv_latent_dim: 192 # Compressed KV dimension (instead of hidden_size)
  q_latent_dim: 192 # Compressed query dimension
  d_rope: 16 # RoPE dimension
```

### Enabling MoE (Mixture of Experts)

```yaml
model:
  # ... other fields ...

  num_experts: 4 # Total number of experts
  experts_per_tok: 2 # Top-K routing (use 2 to prevent expert collapse)
  shared_expert_enabled: true
  intermediate_size: 1536

trainer:
  load_balance_alpha: 0.01 # MoE load balancing loss weight (required > 0 for MoE)
```

### Hybrid Architectures

Specify which layers use Mamba vs Attention:

```yaml
model:
  # ... other fields ...

  mamba_layers: [0, 1, 2, 4, 5, 6] # These layers use Mamba
  # Other layers use MLA + MoE
```

## Project Structure

```
oxidizr/
├── src/
│   ├── main.rs          # CLI entry point
│   ├── config.rs        # Configuration loading
│   ├── model.rs         # Transformer model
│   ├── mamba.rs         # Mamba1 implementation
│   ├── mamba2.rs        # Mamba2 with SSD
│   ├── mamba3.rs        # Mamba3 with trapezoidal/RoPE/MIMO
│   ├── data.rs          # Data loader
│   └── trainer.rs       # Training loop
├── models/
│   ├── nano-start.yaml  # Educational config
│   ├── nano.yaml        # Hybrid Mamba2 + MLA + MoE
│   ├── nano_mamba3.yaml # Pure Mamba3
│   └── ...              # More examples
├── data/
│   └── nano-start/      # Educational dataset for learning
└── Cargo.toml
```

## System Requirements

### Hardware

- **CPU**: Full training support on CPU - no GPU required. Training will be slower but works completely.
- **GPU**: Recommended for faster training. CUDA 12.x supported (requires `--features cuda`)
- **Memory**: Depends on your model size and batch size

### Software

- Rust 1.70+ ([install via rustup](https://rustup.rs/))
- CUDA Toolkit 12.x (optional, for GPU acceleration)

## Nano Educational Project

The included `nano` configs are part of an educational initiative to help users learn LLM training fundamentals. The philosophy:

- **Zero magic** - Full visibility into the training process
- **Clear examples** - Well-documented configs showing best practices
- **Starting point** - Use as a template for your own experiments

The `data/nano-start/` directory contains a curated dataset designed for learning. It's small enough to train quickly while demonstrating key concepts.

**This is guidance, not a requirement.** Oxidizr is a general-purpose trainer. The nano examples exist to help you get started - you're free to create any architecture and use any dataset you want.

## Tips and Best Practices

### Memory Management

- Start with small batch size and sequence length, then scale up
- Use gradient accumulation to simulate larger batches without OOM
- Monitor VRAM usage (oxidizr estimates memory requirements before training)

### Effective Batch Size

```
effective_batch = batch_size × gradient_accumulation × num_gpus
```

Example: `batch_size=2`, `gradient_accumulation=4`, `num_gpus=2` → effective batch of 16

### Data Prefetching

Enable async data loading to overlap CPU I/O with GPU compute:

```bash
cargo run --release --features cuda -- train -f models/nano.yaml --prefetch 2
```

## Development

```bash
# Run tests
cargo test

# Build documentation
cargo doc --open

# Lint
cargo clippy

# Format
cargo fmt
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Built on [Candle](https://github.com/huggingface/candle) by HuggingFace
- Architecture inspired by Llama 2/3, Mamba, and DeepSeek
- Designed for transparency and ease of use

---

**Status**: Beta | **Version**: 0.1.0 | **Last Updated**: 2025-12-05

## Citation

If you use Splintr in your research, please cite:

```bibtex
@software{splintr,
  author = {Farhan Syah},
  title = {Oxidzr: A Rust-based LLM training framework},
  year = {2025},
  url = {https://github.com/ml-rust/oxidizr}
}
```
