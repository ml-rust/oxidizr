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
| `max_grad_norm` | Global gradient norm clip | 1.0 |
| `max_grad_value` | Per-element gradient clamp, applied before the norm clip. Must be finite and positive. Omit to disable. | none |

### Learning Rate Schedule

Warmup runs first. Step `s` of `W` warmup steps applies `learning_rate * (s + 1) / W`, at least `1e-7`. The schedule covers every later step up to the run's last step `T`. `T` is `max_steps`, capped by the steps in `num_epochs` epochs.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `warmup_steps` | Warmup step count. Overrides `warmup_ratio` | none |
| `warmup_ratio` | Warmup as a fraction of the epoch steps, rounded up | 0.1 |
| `lr_schedule` | `constant`, `cosine`, `linear` or `wsd` | `constant` |
| `min_lr_ratio` | Fraction of `learning_rate` reached at step `T`, in `[0, 1]` | 0.0 |
| `decay_ratio` | `wsd` only: fraction of `T` the decay spans, in `(0, 1]`. Excludes `decay_steps` | 0.1 |
| `decay_steps` | `wsd` only: decay step count. Excludes `decay_ratio` | none |
| `decay_shape` | `wsd` only: `linear`, `cosine` or `sqrt` | `linear` |

Schedules after warmup:

- `constant`: holds `learning_rate`.
- `cosine`, `linear`: decay from `learning_rate` at step `W` to `min_lr_ratio * learning_rate` at step `T`.
- `wsd`: holds `learning_rate` until step `T - D`, then decays over `D` steps to `min_lr_ratio * learning_rate`. `D` is `decay_steps`, else `ceil(decay_ratio * T)`.

Every step from `T` on applies `min_lr_ratio * learning_rate`. A decaying schedule needs `max_steps` or `num_epochs` above 0. A `wsd` run is an error when `W + D` exceeds `T`. The decay keys are an error with any schedule other than `wsd`.

```yaml
trainer:
  learning_rate: 0.0003
  max_steps: 10000
  warmup_steps: 500
  lr_schedule: wsd
  decay_ratio: 0.2      # decay over steps 8000..10000
  decay_shape: sqrt
  min_lr_ratio: 0.1
```

The schedule is a function of the step, so a resumed run continues it at the restored step. The `learning_rate` in headless output and `steps.json` is the rate the logged step applied.

### Weight Decay

AdamW applies decoupled weight decay. A rule picks the parameters it applies to.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `adamw_beta1` | AdamW first-moment coefficient | 0.9 |
| `adamw_beta2` | AdamW second-moment coefficient | 0.95 |
| `adamw_eps` | AdamW epsilon | 1e-6 |
| `weight_decay` | Decay coefficient of the decayed parameters | 0.1 |
| `weight_decay_rule` | `decay_all`, `skip_1d` or `skip_all` | `skip_1d` for a fresh run, the checkpoint's rule for a resume |
| `no_decay_patterns` | Checkpoint-name substrings kept out of decay. `skip_1d` only | none |

Rules:

- `skip_1d`: no decay on tensors of rank below 2, on every tensor whose name ends in `bias`, and on the SSM `a_log` and `D` (`d_param`) tensors. This covers every norm weight, Mamba `dt_bias`, and Mamba1's 2-D `a_log`. Embeddings, LoRA `lora_a`/`lora_b`, and every other weight of rank 2 or more decay.
- `decay_all`: every parameter decays. Runs before this key existed used this rule.
- `skip_all`: no parameter decays.

A pattern matches any part of a checkpoint name, such as `model.layers.0.self_attn.q_proj.weight` or `lm_head.weight`. An empty pattern is an error. Patterns under `decay_all` or `skip_all` are an error. A tied embedding is one tensor, so it counts once.

```yaml
trainer:
  weight_decay: 0.1
  weight_decay_rule: skip_1d
  no_decay_patterns: [embed_tokens]   # also keep the embedding out of decay
```

The checkpoint meta records the rule, and a resume inherits each unset key. A meta without a rule reads as `decay_all`, so a run started before this key resumes bit for bit. A set key that differs from the checkpoint's rule prints a warning, and the resume is not exact. Under `--headless` it is an error. The run logs the rule once at start, with the tensor and parameter count of each side. Headless mode writes that line to stderr.

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
