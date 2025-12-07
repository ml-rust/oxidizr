# Model Configurations

Example YAML configs for different architectures.

## Available Configs

| Config | Architecture | Description |
|--------|--------------|-------------|
| `nano-start.yaml` | Transformer | Small GPT for learning (cl100k_base vocab) |
| `nano.yaml` | Hybrid Mamba2 + MLA + MoE | Production-ready hybrid |
| `nano_mamba2.yaml` | Hybrid Mamba2 + MLA | Mamba2 with attention layers |
| `nano_mamba3.yaml` | Pure Mamba3 | All Mamba3 layers |
| `nano_mamba3_hybrid.yaml` | Hybrid Mamba3 + MLA + MoE | Mamba3 with attention layers |
| `nano_mamba3_mimo.yaml` | Pure Mamba3 MIMO | Mamba3 with multi-input/output |

## Quick Start

```bash
# Educational (recommended for beginners)
cargo run --release -- -f models/nano-start.yaml -d data/nano-start/tokenized/combined.bin

# Production hybrid
cargo run --release -- -f models/nano.yaml -d path/to/data.bin

# Pure Mamba3
cargo run --release -- -f models/nano_mamba3.yaml -d path/to/data.bin
```

## Architecture Overview

### Transformer (nano-start.yaml)

Standard GPT-style transformer for learning:

```yaml
model:
  hidden_size: 256
  num_layers: 6
  num_heads: 8
  vocab_size: 100315  # cl100k_base
```

### Mamba2 Hybrid (nano.yaml, nano_mamba2.yaml)

Combines Mamba2 SSM with attention:

```yaml
model:
  mamba_layers: [0, 1, 2, 4, 5, 6]  # Mamba2 layers
  # Layers 3, 7 use MLA + MoE

  mamba2_num_heads: 48
  mamba2_head_dim: 16
  mamba2_state_size: 64
  mamba2_expand: 2
```

### Mamba3 (nano_mamba3*.yaml)

Latest Mamba with three innovations:

```yaml
model:
  mamba3_enabled: true
  mamba3_complex_rope: true   # Complex-valued RoPE
  mamba3_mimo_rank: 0         # 0=SISO, 4=MIMO
  mamba3_use_conv: false      # Trapezoidal discretization

  # Uses Mamba2 base parameters
  mamba2_num_heads: 48
  mamba2_head_dim: 16
  mamba2_state_size: 64
  mamba2_expand: 2
```

## Tokenizer Vocabulary

The `tokenizer_vocab` field is **required** and must match the vocabulary used to tokenize your training data.

```yaml
model:
  vocab_size: 128354
  tokenizer_vocab: llama3  # REQUIRED: must match your training data
```

### Supported Vocabularies

| Name | Description | Vocab Size (with agents) |
|------|-------------|--------------------------|
| `cl100k_base` | GPT-4, GPT-3.5-turbo | ~100,331 |
| `o200k_base` | GPT-4o | ~200,073 |
| `llama3` | Meta Llama 3 family | ~128,354 |
| `deepseek_v3` | DeepSeek V3/R1 | ~128,954 |
| `custom` | Custom vocabulary | varies |

**Note:** `custom` vocabularies are not yet supported in blazr for inference. If you use a custom tokenizer, you'll need to modify blazr's tokenizer module to load your `.tiktoken` file.

### .tiktoken Format

The `.tiktoken` format is a simple text file where each line contains:
```
<base64_token> <rank>
```

Example:
```
SGVsbG8= 0
V29ybGQ= 1
```

Where `SGVsbG8=` decodes to "Hello" (rank 0, highest priority for BPE merging).

## Creating Custom Configs

1. Copy an existing config as template
2. Adjust parameters
3. Set `tokenizer_vocab` to match your training data
4. Ensure constraint: `hidden_size × expand == num_heads × head_dim`

```bash
cargo run --release -- -f path/to/your/config.yaml -d path/to/data.bin
```

## Tips

- Start with `nano-start.yaml` to learn
- Use `--batch-size 1` to test memory usage
- Use `--headless` if progress bar doesn't appear
