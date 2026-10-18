<div align="center">

# oxidizr

<h3>LLM training from a YAML config, in Rust.</h3>

<p>
Trains GPT-style language models from a YAML config and a tokenized dataset.
Supports dense GQA transformers, Mamba1, Mamba2, Mamba3, MLA, and MoE layers
on CPU or CUDA, built on <a href="https://github.com/ml-rust/boostr">boostr</a>.
</p>

<p>
  <a href="https://docs.rs/oxidizr"><strong>Docs</strong></a>
  ·
  <a href="https://crates.io/crates/oxidizr"><strong>Crate</strong></a>
  ·
  <a href="#bring-your-own-config-and-data"><strong>Config</strong></a>
  ·
  <a href="#cli-reference"><strong>CLI</strong></a>
  ·
  <a href="#quick-start"><strong>Example</strong></a>
  ·
  <a href="CONTRIBUTING.md"><strong>Contributing</strong></a>
</p>

<p>
  <a href="https://discord.gg/jBhFk9kHPg">
    <img src="https://img.shields.io/discord/1453357769720594543?label=Discord&logo=discord&logoColor=white&color=5865F2" alt="Join the Discord">
  </a>
</p>

<p>
  <a href="https://crates.io/crates/oxidizr">
    <img src="https://img.shields.io/crates/v/oxidizr" alt="crates.io">
  </a>
  <a href="https://docs.rs/oxidizr">
    <img src="https://img.shields.io/docsrs/oxidizr" alt="docs.rs">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/crates/l/oxidizr" alt="License">
  </a>
  <a href="https://github.com/ml-rust/oxidizr/stargazers">
    <img src="https://img.shields.io/github/stars/ml-rust/oxidizr?style=social" alt="GitHub stars">
  </a>
</p>

</div>

## What is oxidizr

oxidizr trains a language model from two inputs:

1. **A YAML config** that defines the architecture, optimizer, and training schedule.
2. **A tokenized dataset** stored as a binary file of `u32` tokens.

oxidizr runs the training loop, optimizer, checkpointing, and metrics. It writes
checkpoints as `.safetensors` files and packages them for distribution.

| Layer | Config keys | Notes |
|-------|-------------|-------|
| Dense GQA transformer | `kv_heads`, with `kv_latent_dim` unset | Grouped-query attention with RoPE and RMSNorm |
| MLA (Multi-Head Latent Attention) | `kv_latent_dim`, `q_latent_dim`, `d_rope` | Compressed KV cache |
| Mamba1 | `mamba_layers`, with `mamba2_num_heads` unset | Selective state-space layers |
| Mamba2 | `mamba_layers`, `mamba2_*` | State Space Duality (SSD) |
| Mamba3 | `mamba3_enabled: true`, `mamba2_*` | Trapezoidal discretization. Optional `mamba3_complex_rope`, `mamba3_mimo_rank` |
| MoE (Mixture of Experts) | `num_experts`, `experts_per_tok`, `shared_expert_enabled` | Top-k routing with a load-balance loss |

Layers mix within one model. `mamba_layers` lists the SSM layer indices, and every other layer uses attention.

Mamba2 and Mamba3 require `hidden_size * mamba2_expand == mamba2_num_heads * mamba2_head_dim`. For example, `384 * 2 == 48 * 16`.

## Installation

oxidizr depends on `boostr` through the path `../boostr`. Clone both repositories into the same parent directory:

```bash
git clone https://github.com/ml-rust/boostr
git clone https://github.com/ml-rust/oxidizr
cd oxidizr
cargo build --release
```

Add `--features cuda` to the build for GPU support. The CUDA build uses the toolkit installed on the system.

## Quick Start

Build the project, download the educational dataset, and start training:

```bash
git clone https://github.com/ml-rust/boostr
git clone https://github.com/ml-rust/oxidizr
cd oxidizr
cargo build --release

pip install huggingface_hub
hf download fs90/nano-start-data-bin --local-dir data/nano-start/tokenized --repo-type dataset

cargo run --release -- train -f models/nano-start/nano-start.yaml -d data/nano-start/tokenized/combined.bin
```

The default build trains on CPU. Checkpoints for this config go to `./checkpoints/nano-start`.

To train on a GPU, build and run with `--features cuda`:

```bash
cargo run --release --features cuda -- train -f models/nano-start/nano-start.yaml -d data/nano-start/tokenized/combined.bin
```

After training, package the model. See [Model Distribution](#model-distribution).

## Bring Your Own Config and Data

### Write a Config

A config has two top-level keys: `model` and `trainer`. This dense GQA example is a complete starting point:

```yaml
# my-model.yaml
model:
  name: my-model
  hidden_size: 512
  num_layers: 8
  num_heads: 8
  kv_heads: 4              # Fewer KV heads than query heads (GQA)
  intermediate_size: 2048
  vocab_size: 100315
  tokenizer_vocab: cl100k_base   # Must match the tokenizer that produced the data
  max_seq_len: 512
  rope_theta: 10000.0

trainer:
  target_device: gpu
  learning_rate: 0.0003
  batch_size: 2
  gradient_accumulation: 1
  max_steps: 5000
  num_epochs: 2
  checkpoint_dir: "./checkpoints/my-model"
  log_interval: 10
  save_interval: 500
  max_checkpoints: 3
  load_balance_alpha: 0.0  # MoE auxiliary loss weight. Required. 0.0 disables it.
```

Run it:

```bash
cargo run --release -- train -f my-model.yaml -d data/my-dataset.bin
```

`tokenizer_vocab` is required. Set it to `cl100k_base`, `o200k_base`, `llama3`, `deepseek_v3`, or `custom`.

Key `trainer` fields:

| Field | Purpose |
|-------|---------|
| `learning_rate`, `batch_size`, `gradient_accumulation` | Optimizer step size and batch shape |
| `max_steps`, `num_epochs` | Training length |
| `warmup_steps`, `lr_schedule`, `min_lr_ratio` | Warmup, then `constant`, `cosine`, `linear` or `wsd` decay. Logged rates are the rates each step applied |
| `weight_decay`, `weight_decay_rule`, `no_decay_patterns` | AdamW decay coefficient, and the parameters it applies to. `skip_1d` (fresh-run default) skips norms, biases and SSM `a_log`/`D`. `decay_all` decays everything. `skip_all` decays nothing. A resume keeps the checkpoint's rule |
| `checkpoint_dir`, `save_interval`, `max_checkpoints` | Checkpoint location, frequency, and retention (`0` keeps all) |
| `log_interval` | Steps between metric updates |
| `target_device` | `gpu` or `cpu`. The CLI `--target-device` overrides it |
| `load_balance_alpha` | MoE load-balance loss weight |

The full key reference is in [docs/configuration.md](docs/configuration.md).

### Sample Configs

Start from a file in `models/`. [models/README.md](models/README.md) describes each one. `models/nano-start/nano-start.yaml` is the educational hybrid model (Mamba2 + MLA + MoE).

### Prepare the Data

oxidizr memory-maps a binary file of native-endian `u32` tokens. The tokenizer must match `model.tokenizer_vocab`.

This Python snippet uses [splintr](https://github.com/ml-rust/splintr) to write a token file:

```python
from splintr import Tokenizer
import numpy as np

tokenizer = Tokenizer("llama3")
tokens = tokenizer.encode("Your training text here...")
np.array(tokens, dtype=np.uint32).tofile("data/my-dataset.bin")
```

Pass the file with `-d`. Without `-d`, `train` generates 1,000,000 random tokens for a smoke test.

The `oxidizr::data::create_dummy_data(vocab_size, total_tokens)` function returns the same random tokens as a `Vec<u32>`.

## CLI Reference

```bash
oxidizr <SUBCOMMAND> [OPTIONS]
```

| Subcommand | Purpose | Feature |
|------------|---------|---------|
| `train` | Train a model. `oxidizr -f <config>` runs `train` without the subcommand | none |
| `import` | Convert a HuggingFace checkpoint to an oxidizr checkpoint (`-f`, `-m`, `-d`, `-o`) | none |
| `merge` | Fold LoRA adapters into base weights (`-f`, `-c`, `-o`) | none |
| `pack` | Package a checkpoint for distribution | none |
| `push` | Upload a packaged model to the HuggingFace Hub | `huggingface` |
| `prep-corpus` | Build a packed speech-LM corpus from audio recordings | `audio` |

Common `train` flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `-f, --config <FILE>` | `models/nano.yaml` | YAML config |
| `-d, --data <FILE>` | 1,000,000 random tokens | Memory-mapped `u32` token file |
| `--target-device <gpu\|cpu>` | Config `trainer.target_device` | Device. A build without `cuda` runs on CPU |
| `--batch-size`, `--seq-len`, `--grad-accum`, `--max-steps` | Config | Override the config value |
| `--resume <PATH\|auto>` | None | Resume from a `.safetensors` checkpoint, or the latest one. Restores weights, optimizer state and data position |
| `--gpus <ID>` | `0` | One GPU ID. More than one ID is an error. `CUDA_VISIBLE_DEVICES` selects the device |
| `--prefetch <N>` | `0` | Batches to prefetch on a background thread |
| `--max-checkpoints <N>` | `10` | Checkpoints to keep. `0` keeps all |
| `--dtype <f32\|f16\|bf16>` | Config `model.dtype` (`f32`) | Model precision |
| `--checkpoint-dir <DIR>` | Config `trainer.checkpoint_dir` | Checkpoint directory |
| `--headless` | Off | Print JSON metrics only. Use in CI, pipes, and non-interactive shells |
| `--run-dir <DIR>` | `$OXIDIZED_RUN_DIR`, else `./runs/<timestamp>` | Directory for `run.json` and `steps.json` |

Every run writes `run.json` to its run directory when it ends, in both output modes. It records `status` (`completed` or `failed`), the error text of a failed run, the final step, the last loss, and start and end times. `train` exits with code 1 when the run failed. `steps.json` holds one JSON line per log interval and one for the first step of each session.

Without `--headless`, `train` shows a progress bar. Run `oxidizr <SUBCOMMAND> --help` for every flag. [docs/cli.md](docs/cli.md) documents each subcommand.

## Model Distribution

Package a checkpoint, then upload it:

```bash
oxidizr pack --checkpoint latest      # Or: final, or a step number. Omit to pick interactively
cargo build --release --features huggingface
pip install huggingface_hub
oxidizr push --model hf/<username>/<model-name>
```

`pack` writes `model.safetensors`, `config.json`, and a generated `README.md` to `hf/<username>/<model>/`. It also copies `training_config.json` when the checkpoint has a `run.json`.

`push` runs the `hf` CLI, so `hf` must be on `PATH`. It accepts `--private`, `-t, --token`, and `-y, --yes`.

Put these values in a `.env` file, copied from `.env.example`:

```env
HF_USERNAME=your-username
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Get a token from the [HuggingFace token page](https://huggingface.co/settings/tokens). See [hf/README.md](hf/README.md) and [docs/cli.md](docs/cli.md) for the full flag list.

## Multi-GPU Training

Multi-GPU training is not available. Pass a single GPU:

```bash
cargo run --release --features cuda -- train -f models/nano.yaml --gpus 0
```

`--gpus 0,1` exits with an error. Use `CUDA_VISIBLE_DEVICES` to pick which GPU runs.

## Features

| Feature | Default | Enables |
|---------|---------|---------|
| `f16` | On | F16 and BF16 dtypes |
| `cuda` | Off | CUDA backend (boostr CubeCL) |
| `nccl` | Off | Enables `boostr/nccl`. Implies `cuda`. Adds no training feature |
| `huggingface` | Off | `push` command |
| `audio` | Off | `prep-corpus` command, with the voxr corpus pipeline and the Whisper, NeuCodec, and Silero VAD models |
| `vocab-<name>` | Off | Bundled tokenizer vocabulary for `prep-corpus --tokenizer <name>` |
| `vocabs` | Off | Every bundled vocabulary |

Bundled vocabulary names: `cl100k`, `o200k`, `gpt-oss`, `phi4`, `olmo2`, `llama3`, `llama2`, `deepseek`, `qwen`, `glm`, `gemma2`, `gemma3`, `gemma4`, `kimi`, `mistral`, `modernbert`.

Example: `cargo build --release --features cuda,huggingface`.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for build, test, and lint commands, and the project layout.

Documentation: [docs/README.md](docs/README.md) indexes the guides. [docs/configuration.md](docs/configuration.md) is the config reference. [docs/cli.md](docs/cli.md) is the CLI reference. [docs/architecture/overview.md](docs/architecture/overview.md) describes the design.

## Citation

```bibtex
@software{oxidizr,
  author = {Farhan Syah},
  title = {oxidizr: A Rust-based LLM training framework},
  year = {2025},
  url = {https://github.com/ml-rust/oxidizr}
}
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions and checks.

Questions and design discussion happen on [Discord](https://discord.gg/jBhFk9kHPg).

## License

Licensed under Apache-2.0. See [LICENSE](LICENSE).
