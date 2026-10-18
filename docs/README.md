# Oxidizr Documentation

Welcome to the Oxidizr documentation. Oxidizr is a production-grade LLM training framework written in Rust.

## Getting Started

- [Quick Start Guide](quickstart.md) - Train your first model in 5 minutes
- [CLI Reference](cli.md) - Complete command-line options
- [Configuration Guide](configuration.md) - YAML config file format

## Architecture

- [Architecture Overview](architecture/overview.md) - Supported model architectures
- [Mamba Guide](architecture/mamba.md) - Mamba1, Mamba2, and Mamba3 state space models
- [MLA Guide](architecture/mla.md) - Multi-Head Latent Attention
- [MoE Guide](architecture/moe.md) - Mixture of Experts
- [GDN Guide](architecture/gdn.md) - Gated DeltaNet (Qwen3.5 / Qwen3-Next linear attention)

## Training

- [CPU Training](training/cpu.md) - Training without GPU
- [GPU Training](training/gpu.md) - CUDA acceleration
- [Checkpoints](training/checkpoints.md) - Saving and resuming training

## Data

- [Data Format](data/format.md) - Binary token format
- [Tokenization](data/tokenization.md) - Using splintr for tokenization
- [Educational Dataset](data/nano-start.md) - The nano-start learning dataset

## Guides

- [Common Issues](guides/troubleshooting.md) - FAQ and solutions
- [Output Modes](guides/output-modes.md) - TUI vs headless mode
