# Configuration Guide

Oxidizr uses YAML configuration files to define model architecture and training settings.

## Basic Structure

```yaml
model:
  # Model architecture parameters
  hidden_size: 512
  num_layers: 8
  # ... more model params

trainer:
  # Training hyperparameters
  learning_rate: 0.0003
  batch_size: 2
  # ... more training params
```

## Model Parameters

### Core Architecture

| Parameter | Description | Example |
|-----------|-------------|---------|
| `hidden_size` | Model dimension | 512 |
| `num_layers` | Number of transformer/mamba layers | 8 |
| `num_heads` | Attention heads | 8 |
| `kv_heads` | KV heads for GQA (≤ num_heads) | 4 |
| `vocab_size` | Vocabulary size | 128354 |
| `max_seq_len` | Maximum sequence length | 512 |
| `intermediate_size` | FFN hidden dimension | 2048 |

### Position Encoding

| Parameter | Description | Default |
|-----------|-------------|---------|
| `rope_theta` | RoPE base frequency | 10000.0 |

### Mamba2 Parameters

Enable Mamba2 by adding these parameters:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `mamba2_num_heads` | SSM heads | 48 |
| `mamba2_head_dim` | Dimension per head | 16 |
| `mamba2_state_size` | State dimension | 64 |
| `mamba2_chunk_size` | Chunk size for SSD | 64 |
| `mamba2_n_groups` | Number of groups | 1 |
| `mamba2_conv_kernel` | Conv kernel size | 4 |
| `mamba2_expand` | Expansion factor | 2 |
| `mamba_layers` | Which layers use Mamba | [0,1,2,4,5,6] |

**Constraint:** `hidden_size * mamba2_expand == mamba2_num_heads * mamba2_head_dim`

### MLA Parameters

Enable MLA (Multi-Head Latent Attention):

| Parameter | Description | Example |
|-----------|-------------|---------|
| `kv_latent_dim` | Compressed KV dimension | 192 |
| `q_latent_dim` | Compressed query dimension | 192 |
| `d_rope` | RoPE dimension | 16 |

### MoE Parameters

Enable Mixture of Experts:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `num_experts` | Total expert networks | 4 |
| `experts_per_tok` | Experts per token (top-K) | 2 |
| `shared_expert_enabled` | Enable shared expert | true |

## Trainer Parameters

### Basic Training

| Parameter | Description | Default |
|-----------|-------------|---------|
| `learning_rate` | Learning rate | 0.0003 |
| `batch_size` | Batch size | 2 |
| `max_steps` | Maximum training steps | 5000 |
| `num_epochs` | Number of epochs | 2 |
| `gradient_accumulation` | Gradient accumulation steps | 1 |
| `shuffle_rows` | Visit rows in a per-epoch permutation seeded from `seed` and the epoch; `false` = stream order | true |

### Checkpointing

| Parameter | Description | Default |
|-----------|-------------|---------|
| `checkpoint_dir` | Checkpoint directory | "./checkpoints" |
| `save_interval` | Steps between saves | 500 |
| `log_interval` | Steps between logs | 10 |

### Advanced

| Parameter | Description | Default |
|-----------|-------------|---------|
| `load_balance_alpha` | MoE load balancing weight | 0.0 |

## Complete Example

### Standard Transformer

```yaml
model:
  hidden_size: 512
  num_layers: 8
  num_heads: 8
  kv_heads: 4
  vocab_size: 128354
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

### Hybrid Mamba2 + MLA

```yaml
model:
  hidden_size: 384
  num_layers: 8
  num_heads: 6
  kv_heads: 2
  vocab_size: 128354
  max_seq_len: 512
  rope_theta: 10000.0
  intermediate_size: 1536

  # Mamba2
  mamba2_num_heads: 48
  mamba2_head_dim: 16
  mamba2_state_size: 64
  mamba2_chunk_size: 64
  mamba2_n_groups: 1
  mamba2_conv_kernel: 4
  mamba2_expand: 2
  mamba_layers: [0, 1, 2, 4, 5, 6]

  # MLA
  kv_latent_dim: 192
  q_latent_dim: 192
  d_rope: 16

trainer:
  learning_rate: 0.002
  batch_size: 2
  max_steps: 5000
  num_epochs: 2
  gradient_accumulation: 1
  checkpoint_dir: "./checkpoints"
  log_interval: 10
  save_interval: 500
```

### With MoE

```yaml
model:
  # ... base config ...
  num_experts: 4
  experts_per_tok: 2
  shared_expert_enabled: true

trainer:
  # ... base config ...
  load_balance_alpha: 0.01  # Required > 0 for MoE
```

## CLI Overrides

You can override config values from the command line:

```bash
cargo run --release -- -f config.yaml \
    --batch-size 4 \
    --seq-len 256 \
    --max-steps 1000 \
    --grad-accum 2
```

See [CLI Reference](cli.md) for all override options.
