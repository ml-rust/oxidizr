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

### Gated DeltaNet Parameters

Gated DeltaNet (GDN) is the linear-attention layer of Qwen3.5 and Qwen3-Next. `gdn_layers` picks the GDN layers. Every layer in neither `gdn_layers` nor `mamba_layers` is an attention layer. See [architecture/gdn.md](architecture/gdn.md).

| Parameter | Description | Default |
|-----------|-------------|---------|
| `gdn_layers` | Layer indexes that are GDN layers | none |
| `gdn.state_size` | Head width `S` of q, k and v. Each value head keeps an `[S, S]` state | required |
| `gdn.key_heads` | Query/key heads | required |
| `gdn.value_heads` | Value heads. Must be a multiple of `gdn.key_heads` | required |
| `gdn.conv_kernel` | Causal conv width over the qkv projection | 4 |
| `gdn.chunk_size` | Chunk length of the delta-rule recurrence | 64 |
| `gdn.rms_eps` | Epsilon of the q/k L2 norm and the gated output norm | 1e-6 |

```yaml
model:
  gdn_layers: [0, 1, 2, 4, 5, 6]
  gdn:
    state_size: 64
    key_heads: 2
    value_heads: 4
```

Checks, each a config error that names the key:

- `gdn_layers` needs `gdn`, and `gdn` needs a non-empty `gdn_layers`.
- Each index is below `num_layers`, listed once, and absent from `mamba_layers`.
- Every `gdn` count is above 0, and `rms_eps` is finite and above 0.
- `rope_scaling`, `sliding_window` and `qk_norm` need at least one attention layer.

Each GDN layer also holds `input_layernorm`, `post_attention_layernorm` and the FFN: an MLP, or MoE when `num_experts` is set.

LoRA reaches the mixer through `lora.target_modules`: `in_proj_qkv`, `in_proj_z`, `alpha_proj`, `beta_proj` and `out_proj`. A GDN target without `gdn_layers` is a config error. The MLP targets also adapt the GDN layers' MLP.

Both keys are omitted from the hashed config while absent, so existing checkpoints keep their config hash.

## Trainer Parameters

### Basic Training

| Parameter | Description | Default |
|-----------|-------------|---------|
| `learning_rate` | Learning rate | 0.0003 |
| `batch_size` | Batch of one GPU. A step reads `batch_size x GPUs x gradient_accumulation` rows | 2 |
| `max_steps` | Maximum training steps | 5000 |
| `num_epochs` | Number of epochs | 2 |
| `gradient_accumulation` | Gradient accumulation steps | 1 |
| `shuffle_rows` | Visit rows in a per-epoch permutation seeded from `seed` and the epoch; `false` = stream order | true |

#### Loss weighting under gradient accumulation

A run without a loss mask takes the mean loss of each micro-batch and scales it by `1 / gradient_accumulation`.

A masked run (`trainer.target_mask`, `model.speech_layout` or `trainer.document_mask`) weights every scored token of the step equally:

- The trainer draws all `gradient_accumulation` micro-batches of the step before the first forward pass.
- Each loader counts the scored positions from its host copy of the mask. No device read is needed.
- Each micro-batch adds its summed masked loss times `1 / total`. `total` counts the scored tokens of the whole step.
- A micro-batch with no scored token adds no cross-entropy term. Its MoE auxiliary losses still apply.
- A step with no scored token and no auxiliary loss reports loss 0 and applies no gradient.

The reported loss is the per-token mean over the step. MoE auxiliary losses keep the `1 / gradient_accumulation` weight in both modes.

The unmasked path keeps the old mean form on purpose. The sum form gives the same value in exact arithmetic, but its float rounding differs. The mean form keeps unmasked runs, and their resumes, bit-identical to earlier releases.

### Checkpointing

| Parameter | Description | Default |
|-----------|-------------|---------|
| `checkpoint_dir` | Checkpoint directory | "./checkpoints" |
| `save_interval` | Steps between saves | 500 |
| `max_checkpoints` | Interval sets to keep. 0 keeps every set | 10 |
| `async_checkpoint` | Commit checkpoint sets on a background thread | false |
| `log_interval` | Steps between logs | 10 |

`async_checkpoint: true` moves the file writes off the training thread:

- At each save the training thread copies the set to host memory, then runs the next step.
- One writer thread writes the optimizer file, the weights, then the meta. The weights and optimizer files are byte-equal to a run without the key. The meta differs only in its timestamp and directory path.
- At most one save is in flight. A save that finds the previous one unfinished waits for it, and the run prints one warning.
- A failed write stops the run at the next step boundary. The error names the label, the step and the directory.
- The final set is written on the training thread, after every earlier save commits.
- The key has effect only with `save_interval` above 0. In a data-parallel run, rank 0 alone holds the writer.
- `--async-checkpoint` sets the key to true for one run.

### Advanced

| Parameter | Description | Default |
|-----------|-------------|---------|
| `load_balance_alpha` | MoE load balancing weight | 0.0 |
| `max_grad_norm` | Global gradient norm clip | 1.0 |
| `max_grad_value` | Per-element gradient clamp, applied before the norm clip. Must be finite and positive. Omit to disable. | none |

### Loss Scaling (f16)

f16 has a 5-bit exponent, so gradients below about `6e-8` flush to zero. An f16 run scales its loss to keep them. bf16 and f32 share f32's exponent range and need no scaling.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `loss_scale.initial` | Starting scale. Finite and positive | 65536 |
| `loss_scale.growth_factor` | Scale multiplier after `growth_interval` clean steps. Finite and above 1 | 2 |
| `loss_scale.backoff_factor` | Scale multiplier on an overflow step. Above 0 and below 1 | 0.5 |
| `loss_scale.growth_interval` | Clean steps in a row before the scale grows. At least 1 | 2000 |

Rules:

- `model.dtype: f16` always scales. Without the `trainer.loss_scale` key, the run uses the defaults above. A partial block fills the other keys with their defaults.
- `trainer.loss_scale` under `bf16` or `f32` is an error that names the key and the dtype. The check uses the final dtype, after `--dtype`.
- The trainer multiplies each backward loss by the scale, on top of the `1 / gradient_accumulation` or `1 / total` weighting.
- After accumulation, every gradient widens to f32 and is divided by the scale. `max_grad_value`, `max_grad_norm` and `target_grad_norm` then apply to the unscaled gradients.
- A non-finite gradient norm is an overflow. The step skips the optimizer, so the parameters and the AdamW step count stay unchanged. The scale backs off, and a stderr line reports it. The step still draws its batches, so the data position stays valid.
- The loss and `grad_norm` in logs and `steps.json` are unscaled. An overflow step reports a non-finite `grad_norm`.

```yaml
model:
  dtype: f16
trainer:
  loss_scale:
    initial: 65536
    growth_interval: 1000   # other keys keep their defaults
```

The checkpoint meta records the scale, so a resume continues it exactly. See [Resume Behavior](training/checkpoints.md#resume-behavior).

### Training Data

`trainer.data.sources` lists the token files of a run. The run reads them as one row stream, in list order. Repeated `-d/--data` flags name the same list.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `data.sources[i].path` | Token file `i`. `${VAR}` expands | none |
| `data.sources[i].weight` | Share of each epoch's rows for file `i`, relative to the other weights. Finite and above 0 | none |
| `data.rows_per_epoch` | Rows per epoch of a weighted mixture. Needs weights | every training row of every source |

- Setting `trainer.data.sources` and `-d` together is an error naming both.
- An empty list, an empty path or an unknown key is an error.
- Each file drops its own tail shorter than a row. No row spans two files.
- One source trains bit-identical to `-d` with that file.
- Set `weight` on every source, or on none. A weight on some sources only is an error naming both keys.
- A weight of 0, below 0, or not finite is an error. `rows_per_epoch: 0` is an error.
- Without weights the run reads one stream in list order, as before.
- With weights, file `i` supplies `c_i` rows of each epoch of `L = rows_per_epoch` rows. `c_i` is the largest-remainder rounding of `w_i / sum(w) * L`. A tie goes to the lower index.
- The holdout comes off each file first. The mixture covers the training rows only, and the eval stays unmixed.
- A resume with other weights or another `rows_per_epoch` is an error naming the key. Resume with the original config, or start a fresh run.
- See [Weighted mixture](data/format.md#weighted-mixture) for the slot order, upsampling and resume rules.

```yaml
trainer:
  data:
    sources:
      - path: data/web.bin
      - path: ${DATA_DIR}/code.bin
```

A weighted mixture of the same files:

```yaml
trainer:
  data:
    rows_per_epoch: 100000
    sources:
      - path: data/web.bin
        weight: 0.7
      - path: ${DATA_DIR}/code.bin
        weight: 0.3
```

### Validation and Early Stopping

`trainer.eval` scores held-out data at step boundaries. Without the key the run evaluates nothing and trains as before.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `eval.data_file` | Token file the eval reads in full. Training keeps every row of its data | none |
| `eval.holdout_rows` | Whole rows split off the end of the training data. One data source only | none |
| `eval.holdout_fraction` | Fraction of each data source's rows split off its end, in `(0, 1)`. Rounds down per source | none |
| `eval.interval` | Steps between evals. Positive | 500 |
| `eval.batches` | Batches per eval. Positive. Capped at the whole batches the eval rows hold | 50 |
| `eval.early_stopping.patience` | Evals in a row without an improvement that stop the run. Positive | 5 |
| `eval.early_stopping.min_delta` | An eval improves only when its loss is below `best - min_delta`. Finite, 0 or more | 0.0 |

Rules:

- Without `data_file`, set exactly one of `holdout_rows` and `holdout_fraction`. With `data_file`, set neither.
- Each error names the key it rejects. An unknown key inside `eval` is an error.
- A holdout of `h` rows from `N`: training reads rows `[0, N - h)` and the eval reads `[N - h, N)`. A held-out row never reaches a training batch. Both sides must hold one batch of `batch_size` rows.
- Over several data sources, each source holds out `holdout_fraction` of its own rows off its end. The eval reads the held-out ranges in source order. `holdout_rows` with more than one source is an error naming `holdout_fraction`.
- An eval reads its rows in stream order from the first row, with no shuffle, so every eval sees the same batches. It owns its own loader and never moves the training loader or its row order.
- An eval runs when the completed step count is a multiple of `interval`, and once after the last step. It runs before the interval checkpoint of the same step.
- The eval batches use `batch_size` rows and the training loss mask. The forward pass runs without a graph. The eval loss is the cross-entropy summed over every scored token of every eval batch, divided by the scored token count. MoE auxiliary and z-losses stay out. Perplexity is `exp(loss)`.
- An eval loss below `best - min_delta` is the new best. Without `early_stopping` the run tracks the best and never stops. The first finite loss is always a best. A non-finite loss never is.
- A new best writes the `best` checkpoint when `save_interval` is above 0. See [Checkpoints](training/checkpoints.md#best-checkpoint).
- After `patience` evals in a row without a best, the run stops at that step boundary. It writes the final checkpoint and exits 0. `run.json` records `"status": "completed"`, `"stopped_early": true` and a `stop_reason`.

```yaml
trainer:
  eval:
    holdout_fraction: 0.02   # last 2% of the rows
    interval: 500
    batches: 50
    early_stopping:
      patience: 5
      min_delta: 0.001
```

The checkpoint meta records the held-out row count and the stop counters. A resume continues the counters exactly. A resume with another held-out row count is an error, because the count decides which rows the saved data position counts.

### Throughput and MFU

Every step reports its token count and tokens per second. `trainer.metrics` adds model FLOPs utilization (MFU).

| Parameter | Description | Default |
|-----------|-------------|---------|
| `metrics.peak_tflops` | Peak dense TFLOP/s of one device at the run's precision. Finite and positive. Absent: no MFU | none |

Rules:

- `tokens` is the step's rows over every rank times `max_seq_len`: `batch_size * gradient_accumulation * world * max_seq_len`. A masked run also reports `scored_tokens`, the scored targets over every rank.
- `tokens_per_sec` is the session's tokens over every rank divided by its elapsed seconds.
- `MFU = flops_per_token * tokens_per_sec / (world * peak_tflops * 1e12)`. `peak_tflops` is per device. oxidizr never guesses it.
- `flops_per_token = 6 * N + 12 * L * W * T`, the PaLM form:
  - `N` counts every parameter of rank 2 or more. Norm scales, biases and the SSM `A`/`D` vectors stay out.
  - The input embedding is a gather and stays out. The output head counts once, tied or not.
  - A routed MoE expert counts `experts_per_tok / num_experts` of its weights. The router and the shared expert count in full.
  - A frozen weight, such as a LoRA base weight, counts 4 per token instead of 6: it has no weight gradient.
  - `L` is the attention layers, `W` is `num_heads * head_dim`, and `T` is `max_seq_len`, capped at `sliding_window`.
- Limits: the count ignores the causal mask, softmax, norms, activations, the MLA latent width, the Mamba scan and the Gated DeltaNet recurrence. Mamba and Gated DeltaNet layers add only their weights. MFU is therefore a lower bound.

```yaml
trainer:
  metrics:
    peak_tflops: 165.0   # one device's dense bf16 peak, from its datasheet
```

### Loss-Spike Guard

`trainer.spike_guard` flags a step whose loss jumps far above its recent average. Without the key the guard is off, and the run, its meta and its model hash stay as before.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `spike_guard.window` | EMA span in steps. A new loss weighs `2 / (window + 1)`. At least 1 | 100 |
| `spike_guard.z_threshold` | Standard deviations above the EMA mean that flag a spike. Finite and positive | 6.0 |
| `spike_guard.min_steps` | Losses the EMA must hold before the guard flags any step. Values below 2 act as 2 | 50 |
| `spike_guard.action` | `skip` or `warn` | `skip` |

Rules:

- The guard keeps an exponential moving mean and variance of the step loss. It reads the loss after the data-parallel sum, before the optimizer step. Every rank holds the same loss, so every rank flags the same steps.
- A finite loss above `mean + z_threshold * std` is a spike. A spike never enters the EMA. A non-finite loss is never judged and never enters the EMA: it keeps the overflow skip of [Loss Scaling](#loss-scaling-f16).
- `warn` prints a stderr line and writes a `{"spike": ...}` headless line. The step trains as usual.
- `skip` does the same and skips the optimizer step. The parameters and the AdamW step count stay unchanged. The step still draws its batches, so the data position stays valid. With finite gradients the f16 loss scale stays unchanged. A spike step with an overflow backs the scale off as any overflow does.
- `run.json` records `spikes_detected` and `spikes_skipped`. The checkpoint meta records the EMA and the counters under `spike_guard`, so a resume continues exactly.
- An unknown key inside `spike_guard` or `metrics` is an error.

```yaml
trainer:
  spike_guard:
    window: 100
    z_threshold: 6.0
    min_steps: 50
    action: skip
```

### Activation Checkpointing

`trainer.activation_checkpointing` trades compute for memory. A covered layer keeps only its input during the forward pass. The backward pass runs the layer again to rebuild its activations.

| Value | Layers that recompute |
|-------|-----------------------|
| `none` | None. The default |
| `all` | Every layer |
| `{layers: [0, 3, 5]}` | The listed layer indices |

Rules:

- On CPU, the loss, every gradient and the MoE auxiliary and z-losses stay bit-identical to `none`. Tests cover this. On CUDA a test checks agreement within a relative 1e-5. Bitwise equality there is not claimed.
- Peak memory and step time change.
- Each covered layer runs its forward pass twice per step. MFU counts model FLOPs only, so the recompute lowers MFU.
- `layers` must be non-empty, each index below `model.num_layers`, and no index listed twice. Each error names the index, `model.num_layers` and the key `trainer.activation_checkpointing.layers`.
- The key stays out of the model hash and the checkpoint meta. A resume can change it.
- Every data-parallel rank uses the same value. Evals never recompute: their forward pass keeps no graph.
- Every layer type supports it: Mamba1, Mamba2, Mamba3, MLA, GQA, Gated DeltaNet, with an MLP or MoE FFN, LoRA included.

```yaml
trainer:
  activation_checkpointing: all
  # or: activation_checkpointing: {layers: [0, 1, 2, 3]}
```

### Document Masking

`trainer.document_mask` stops attention, recurrent state and loss at document boundaries inside a packed row. [Document Boundaries](data/format.md#document-boundaries) gives the segment and loss rules.

| Key | Description | Default |
|-----|-------------|---------|
| `separator_id` | The id that ends every document | The `separator_id` of the data's `.tokens.json` sidecar |

```yaml
trainer:
  document_mask: {}                  # separator from the data sidecar
  # or: document_mask: {separator_id: 100257}
```

Separator rules:

- An unset `separator_id` comes from the first sidecar of the training files and `trainer.eval.data_file`.
- Every sidecar present must agree with `separator_id` and with each other. The error names both sources and both ids.
- No `separator_id` and no sidecar is an error naming `trainer.document_mask.separator_id`.
- Unknown keys under `document_mask` are an error.

Model rules:

- Mamba, Mamba2, Mamba3 and Gated DeltaNet layers take the mask. Each resets its conv window and recurrent state at a document start, so a document sees nothing of the one before it.
- `model.sliding_window` is accepted. The window counts tokens inside one document, so a document behaves as if it ran alone.
- An MLA model takes the mask in f32, f16 and bf16. Its mask bias uses the model dtype.
- A GQA model on CUDA needs a head dim of 64, 128 or 256. The error names the head dim. The CLI checks this once the target device is final.

Checkpoint and run rules:

- The checkpoint meta records `document_mask: {separator_id}`. A meta without the key is a run without the mask.
- A resume that adds, removes or changes the mask warns. Under `--headless` it is an error.
- Evals use the same mask as training.
- Every data-parallel rank masks its own rows. Each row starts a new document, so ranks need no shared state.
- Recompute under `activation_checkpointing` stays bit-identical on CPU.

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
