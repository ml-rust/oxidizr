# MoE Guide

Mixture of Experts (MoE) increases model capacity by routing each token to a subset of specialized expert networks.

## What is MoE?

Instead of one large FFN, MoE uses multiple smaller "expert" FFNs and a router that selects which experts process each token:

```
Token → Router → Top-K Experts → Weighted sum → Output
```

This provides more parameters (capacity) without proportional compute cost.

## Configuration

```yaml
model:
  # ... base config ...
  num_experts: 4          # Total expert networks
  experts_per_tok: 2      # Top-K routing
  shared_expert_enabled: true
  intermediate_size: 1536

trainer:
  load_balance_alpha: 0.01  # Required > 0 for MoE
```

## Key Parameters

### num_experts

Total number of expert networks. More experts = more capacity.

Typical values: 4, 8, 16, 64

### experts_per_tok

How many experts process each token (top-K routing).

**Important:** Use at least 2 to prevent expert collapse.

- `experts_per_tok: 1` - Single expert per token (can collapse)
- `experts_per_tok: 2` - Recommended minimum
- Higher values = more compute, more stable

### shared_expert_enabled

Whether to include a shared expert that processes all tokens:

- `true` (default when absent or `null`) - Shared expert provides baseline capacity
- `false` - Only routed experts. The checkpoint holds no `moe.shared_expert.*` tensors.

The exported `config.json` always carries the flag explicitly, so it matches the saved weights.
A `config.json` from an older oxidizr holds `null`, while its weights hold a shared expert.
`train` rewrites `config.json` at start, and `pack` writes `true` for such a file.

### load_balance_alpha

Weight for the load balancing auxiliary loss. **Required > 0 for MoE**.

- `0.0` - No load balancing (experts may collapse)
- `0.01` - Typical value
- Higher = stronger push for balanced routing

## Example Configurations

### Basic MoE

```yaml
model:
  hidden_size: 512
  num_layers: 8
  num_heads: 8
  kv_heads: 4
  vocab_size: 128354
  max_seq_len: 512
  rope_theta: 10000.0
  intermediate_size: 1536

  # MoE
  num_experts: 4
  experts_per_tok: 2
  shared_expert_enabled: true

trainer:
  learning_rate: 0.0003
  batch_size: 2
  max_steps: 5000
  load_balance_alpha: 0.01
```

### MoE + MLA

Combine with latent attention:

```yaml
model:
  # ... base config ...

  # MoE
  num_experts: 8
  experts_per_tok: 2
  shared_expert_enabled: true

  # MLA
  kv_latent_dim: 192
  q_latent_dim: 192
  d_rope: 16

trainer:
  load_balance_alpha: 0.01
```

### Hybrid Mamba + MLA + MoE

Full hybrid architecture:

```yaml
model:
  num_layers: 8
  mamba_layers: [0, 1, 2, 4, 5, 6]  # Mamba
  # Layers 3, 7 use MLA + MoE

  # MoE
  num_experts: 4
  experts_per_tok: 2
  shared_expert_enabled: true

  # MLA
  kv_latent_dim: 192
  q_latent_dim: 192
  d_rope: 16

  # Mamba2
  mamba2_num_heads: 48
  # ...

trainer:
  load_balance_alpha: 0.01
```

## Understanding Expert Routing

### Router

The router is a small learned network that outputs probabilities for each expert:

```
router_logits = W_router @ input
expert_probs = softmax(router_logits)
top_k_experts = topk(expert_probs, k=experts_per_tok)
```

### Load Balancing

Without load balancing, training can collapse to using only a few experts. The auxiliary loss encourages balanced usage:

```
balance_loss = load_balance_alpha × variance(expert_usage)
total_loss = main_loss + balance_loss
```

### Expert Collapse

Signs of expert collapse:
- Loss plateaus early
- Some experts never activated
- Training becomes unstable

Prevention:
- Use `experts_per_tok >= 2`
- Set `load_balance_alpha > 0`
- Enable `shared_expert_enabled`

## When to Use MoE

**Good for:**
- Large model capacity needs
- Efficient scaling (more params, less compute)
- Specialized knowledge domains

**Consider alternatives if:**
- Small models (overhead not worth it)
- Latency-critical inference (routing adds overhead)
- Simple tasks

## Performance Considerations

### Memory

MoE uses more memory than dense models:
- `num_experts × expert_size` parameters
- But only `experts_per_tok` experts compute per token

### Compute

Active compute depends on routing:
- Dense equivalent = `experts_per_tok / num_experts` of full MoE
- Example: 2/8 = 25% of parameters active

### Training

MoE can be less stable:
- Start with lower learning rate
- Monitor expert utilization
- Adjust `load_balance_alpha` if needed
