# Architecture Overview

Oxidizr supports multiple architectures for LLM training.

## Base Architectures

### GPT/Llama-style Transformer

The default architecture follows modern Llama-style transformers:

- **RoPE (Rotary Position Embeddings)** - Position information through rotation
- **RMSNorm** - Efficient layer normalization
- **GQA (Grouped Query Attention)** - Fewer KV heads than Q heads for memory efficiency
- **SwiGLU** - Gated activation in FFN layers

```yaml
model:
  hidden_size: 512
  num_layers: 8
  num_heads: 8
  kv_heads: 4        # GQA: 4 KV heads shared across 8 Q heads
  vocab_size: 128354
  max_seq_len: 512
  rope_theta: 10000.0
  intermediate_size: 2048
```

### Mamba (State Space Models)

Mamba models process sequences through selective state spaces instead of attention, enabling efficient long-range context without quadratic complexity.

#### Mamba1

The original Mamba architecture with selective mechanism:

- Linear time complexity O(N) vs O(N²) for attention
- Efficient for very long sequences
- No KV cache needed

```yaml
model:
  # ... base config ...
  mamba_d_state: 16
  mamba_d_conv: 4
  mamba_expand: 2
```

#### Mamba2

Improved Mamba with State Space Duality (SSD) algorithm:

- Faster than Mamba1
- Better parallelization
- Chunk-based processing

```yaml
model:
  # ... base config ...
  mamba2_num_heads: 48
  mamba2_head_dim: 16
  mamba2_state_size: 64
  mamba2_chunk_size: 64
  mamba2_n_groups: 1
  mamba2_conv_kernel: 4
  mamba2_expand: 2

  # CONSTRAINT: hidden_size * expand == num_heads * head_dim
  # Example: 384 * 2 = 768 == 48 * 16
```

#### Mamba3

Latest Mamba variant with improved efficiency.

## Advanced Components

### MLA (Multi-Head Latent Attention)

MLA compresses the KV cache into a lower-dimensional latent space, reducing memory usage during inference while maintaining quality.

```yaml
model:
  # ... base config ...
  kv_latent_dim: 192    # Compressed KV dimension
  q_latent_dim: 192     # Compressed query dimension
  d_rope: 16            # RoPE dimension for position encoding
```

Benefits:
- Smaller KV cache for long sequences
- Faster inference with less memory
- Minimal quality loss vs full attention

See [MLA Guide](mla.md) for details.

### MoE (Mixture of Experts)

MoE routes each token to a subset of specialized expert networks, increasing model capacity without proportional compute cost.

```yaml
model:
  # ... base config ...
  num_experts: 4          # Total expert networks
  experts_per_tok: 2      # Top-K routing (use 2 minimum)
  shared_expert_enabled: true
  intermediate_size: 1536

trainer:
  load_balance_alpha: 0.01  # Required > 0 for MoE
```

Key considerations:
- `experts_per_tok >= 2` prevents expert collapse
- `load_balance_alpha` adds auxiliary loss for balanced routing
- Shared expert provides baseline capacity

See [MoE Guide](moe.md) for details.

## Hybrid Architectures

Oxidizr allows mixing architectures layer-by-layer. Common patterns:

### Mamba2 + MLA (Recommended Hybrid)

Combine Mamba2's efficient sequential processing with MLA's cross-sequence attention:

```yaml
model:
  num_layers: 8
  mamba_layers: [0, 1, 2, 4, 5, 6]  # These use Mamba2
  # Layers 3, 7 use MLA attention
```

This gives you:
- Fast sequential processing (Mamba2 layers)
- Global context when needed (MLA layers)
- Memory efficiency from both

### Mamba2 + MLA + MoE

Full hybrid with expert routing in attention layers:

```yaml
model:
  num_layers: 8
  mamba_layers: [0, 1, 2, 4, 5, 6]
  num_experts: 4
  experts_per_tok: 2
  # Layers 3, 7 use MLA + MoE
```

### Pure Architectures

You can also use pure architectures:

**Pure Mamba2:**
```yaml
model:
  mamba_layers: [0, 1, 2, 3, 4, 5, 6, 7]  # All layers
```

**Pure Transformer:**
```yaml
model:
  # Don't specify mamba_layers
  # Don't specify mamba2_* params
```

## Architecture Selection Guide

| Use Case | Recommended Architecture |
|----------|-------------------------|
| General purpose | Llama-style Transformer |
| Long sequences | Mamba2 or Hybrid |
| Memory constrained | MLA or Mamba |
| Large capacity | MoE |
| Fast training | Mamba2 |
| Maximum quality | Transformer + MoE |

## Sample Configs

Check the `models/` directory for working examples:

| Config | Architecture | Description |
|--------|--------------|-------------|
| `nano.yaml` | Llama Transformer | Standard GPT-style |
| `nano_mamba2.yaml` | Mamba2 + MLA | Hybrid architecture |
| `nano_mamba3.yaml` | Mamba3 | Trapezoidal discretization |
| `nano_mamba3_hybrid.yaml` | Mamba3 + MLA + MoE | Hybrid architecture |

## Implementation Details

### Source Files

| File | Component |
|------|-----------|
| `src/model.rs` | Transformer, attention, FFN, MoE, and the Mamba1/2/3 blocks over boostr's `Mamba1`/`Mamba2`/`Mamba3` |
| `src/config.rs` | Config parsing |
| `src/config_convert.rs` | Conversion to boostr's model config |

### Constraints

1. **Mamba2 dimension constraint:**
   ```
   hidden_size * mamba2_expand == mamba2_num_heads * mamba2_head_dim
   ```

2. **MoE routing:**
   - Use `experts_per_tok >= 2` to prevent collapse
   - Set `load_balance_alpha > 0` for balanced training

3. **Hybrid layers:**
   - `mamba_layers` is a list of layer indices (0-indexed)
   - Unlisted layers use attention (MLA if configured, else standard)
