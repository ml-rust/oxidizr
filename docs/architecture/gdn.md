# Gated DeltaNet (GDN)

Gated DeltaNet is the linear-attention layer of Qwen3.5 and Qwen3-Next. Each value head keeps an `[S, S]` state and updates it with a gated delta rule. Cost is linear in sequence length, and no KV cache grows with it.

oxidizr trains GDN layers through boostr's `GdnBlock`. A model mixes GDN layers, Mamba layers and attention layers by index.

## Config

```yaml
model:
  hidden_size: 384
  num_layers: 8
  num_heads: 6
  kv_heads: 2
  gdn_layers: [0, 1, 2, 4, 5, 6]   # layers 3 and 7 are attention layers
  gdn:
    state_size: 64     # head width S of q, k and v
    key_heads: 2       # query/key heads H_k
    value_heads: 4     # value heads H_v, a multiple of key_heads
    conv_kernel: 4     # default 4
    chunk_size: 64     # default 64
```

`models/nano_gdn_hybrid.yaml` is a complete example. [configuration.md](../configuration.md#gated-deltanet-parameters) lists every key and check.

Derived widths:

| Name | Value |
|------|-------|
| `key_dim` | `key_heads * state_size` |
| `value_dim` | `value_heads * state_size` |
| `qkv_dim` | `2 * key_dim + value_dim` |

## Layer

Each GDN layer has the same frame as an attention layer:

```
x -> input_layernorm -> GDN mixer -> + x -> post_attention_layernorm -> MLP or MoE -> + residual
```

The mixer:

1. `in_proj_qkv`, `in_proj_z`, `alpha_proj` and `beta_proj` project the normed input.
2. A depthwise causal conv with SiLU runs over the qkv projection.
3. q and k are L2-normed. Value head `h` reads key head `h % key_heads` (tiled order).
4. The chunked delta-rule recurrence runs with decay `-exp(A_log) * softplus(alpha + dt_bias)` and step `sigmoid(beta)`.
5. `silu(z) * rms_norm(o)` gates the output, and `out_proj` maps it back to `hidden_size`.

Each training forward starts from a zero conv window and a zero state. One sequence of `max_seq_len` tokens is one backpropagation segment.

An MoE GDN layer adds its load-balance and z losses to the step loss, as an MoE attention layer does.

Activation checkpointing: `trainer.activation_checkpointing` runs a covered GDN layer as one recompute segment, as it does every other layer. On CPU the values stay bit-identical. CUDA bitwise equality is unconfirmed. See [Activation Checkpointing](../configuration.md#activation-checkpointing).

## Checkpoint Layout

Under `model.layers.N.`:

| Name | Shape |
|------|-------|
| `input_layernorm.weight` | `[hidden_size]` |
| `linear_attn.in_proj_qkv.weight` | `[qkv_dim, hidden_size]` |
| `linear_attn.in_proj_z.weight` | `[value_dim, hidden_size]` |
| `linear_attn.alpha_proj.weight` | `[value_heads, hidden_size]` |
| `linear_attn.beta_proj.weight` | `[value_heads, hidden_size]` |
| `linear_attn.out_proj.weight` | `[hidden_size, value_dim]` |
| `linear_attn.conv1d.weight` | `[qkv_dim, 1, conv_kernel]` |
| `linear_attn.A_log` | `[value_heads]` |
| `linear_attn.dt_bias` | `[value_heads]` |
| `linear_attn.norm.weight` | `[state_size]` |
| `post_attention_layernorm.weight` | `[hidden_size]` |
| `mlp.*` or `moe.*` | as on attention layers |

An adapted projection adds `<name>.lora_a` (`[rank, in]`) and `<name>.lora_b` (`[out, rank]`) and keeps `<name>.weight` for its base.

boostr's `GdnBlock::load` reads the `linear_attn.*` names directly. No projection carries a bias.

### Mapping to Other Layouts

| oxidizr (`linear_attn.`) | Hugging Face Qwen3.5 | boostr `qwen35` loader |
|------|------|------|
| `in_proj_qkv.weight` | `in_proj_qkv.weight`, value rows grouped | same name, value rows tiled |
| `in_proj_z.weight` | `in_proj_z.weight`, grouped | same name, tiled |
| `alpha_proj.weight` | `in_proj_a.weight`, grouped | `alpha_proj.weight`, tiled |
| `beta_proj.weight` | `in_proj_b.weight`, grouped | `beta_proj.weight`, tiled |
| `out_proj.weight` | `out_proj.weight`, grouped columns | same name, tiled columns |
| `conv1d.weight` | `conv1d.weight`, value rows grouped | same name, tiled |
| `A_log` | `A_log`, grouped | `a_neg_exp = -exp(A_log)`, tiled |
| `dt_bias` | `dt_bias`, grouped | same name, tiled |
| `norm.weight` | `norm.weight` | same name |

oxidizr trains in the tiled value-head order. Grouped order puts the value heads of key head 0 first. Tiled order interleaves them. An export to Hugging Face must reorder every value-head axis from tiled to grouped, `boostr::format::hf_qwen35::reorder_v_heads` with `key_heads` and `rep` swapped. oxidizr does not run that export.

## Unsupported

- GPU training is unconfirmed. The tests run GDN training, resume and LoRA on CPU only. No run on a real GPU has checked the CUDA path.
- blazr serving. blazr serves GDN through boostr's `qwen35` model. That model also expects Qwen3.5 gated full attention (`q_proj` rows `[query | gate]`, partial rotary). oxidizr's attention layers have no output gate. Training warns and writes no `config.json` for a GDN model, and `oxidizr merge` skips it.
- Qwen3.5 gated attention layers. Every non-GDN, non-Mamba layer is oxidizr's own attention block.
- Import of a Hugging Face Qwen3.5 checkpoint, and export to one. The names and head order differ as the table lists.
- Hadamard-rotated or quantized GDN projections. A trained GDN layer is dense.
