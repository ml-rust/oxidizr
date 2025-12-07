# Quick Start Guide

Train your first model in 5 minutes.

## Prerequisites

- Rust 1.70+ ([install via rustup](https://rustup.rs/))
- Optional: CUDA 12.x/13.x for GPU acceleration

## Step 1: Build

```bash
# Clone the repository
git clone https://github.com/ml-rust/oxidizr.git
cd oxidizr

# Build (CPU)
cargo build --release

# Or build with GPU support
cargo build --release --features cuda
```

## Step 2: Download Dataset

```bash
pip install huggingface_hub
hf download fs90/nano-start-data-bin --local-dir data/nano-start/tokenized --repo-type dataset
```

## Step 3: Train!

```bash
cargo run --release -- -f models/nano-start.yaml -d data/nano-start/tokenized/combined.bin
```

That's it! You should see a progress bar with live metrics.

## Common Options

```bash
# Force CPU (if you have GPU but want CPU)
cargo run --release -- -f models/nano-start.yaml -d data/nano-start/tokenized/combined.bin --target-device cpu

# Limit training steps
cargo run --release -- -f models/nano-start.yaml -d data/nano-start/tokenized/combined.bin --max-steps 1000

# Use headless mode (text output instead of progress bar)
cargo run --release -- -f models/nano-start.yaml -d data/nano-start/tokenized/combined.bin --headless
```

## What's Next?

- [CLI Reference](cli.md) - All command-line options
- [Configuration Guide](configuration.md) - Create your own configs
- [Architecture Overview](architecture/overview.md) - Mamba, MLA, MoE explained
- [Data Format](data/format.md) - Prepare your own dataset

## Troubleshooting

**Progress bar doesn't appear?**

```bash
cargo run --release -- -f models/nano-start.yaml -d data/nano-start/tokenized/combined.bin --headless
```

**Out of memory?**

- Reduce `--batch-size` (try 1)
- Reduce `--seq-len` (try 128)
- Use gradient accumulation in config

**CUDA errors?**

- Ensure CUDA 12.x/13.x is installed
- Build with `--features cuda`
- Check `nvidia-smi` works
