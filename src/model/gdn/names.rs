//! Checkpoint names of a [`GdnLayer`].
//!
//! Under `layers.N`:
//!
//! | Name | Shape |
//! | --- | --- |
//! | `input_layernorm.weight` | `[hidden]` |
//! | `linear_attn.in_proj_qkv.weight` | `[2 * key_dim + value_dim, hidden]` |
//! | `linear_attn.in_proj_z.weight` | `[value_dim, hidden]` |
//! | `linear_attn.alpha_proj.weight` | `[value_heads, hidden]` |
//! | `linear_attn.beta_proj.weight` | `[value_heads, hidden]` |
//! | `linear_attn.out_proj.weight` | `[hidden, value_dim]` |
//! | `linear_attn.conv1d.weight` | `[2 * key_dim + value_dim, 1, conv_kernel]` |
//! | `linear_attn.A_log` | `[value_heads]` |
//! | `linear_attn.dt_bias` | `[value_heads]` |
//! | `linear_attn.norm.weight` | `[state_size]` |
//! | `post_attention_layernorm.weight` | `[hidden]` |
//! | `mlp.*` or `moe.*` | as the attention layers |
//!
//! `key_dim` is `key_heads * state_size`, `value_dim` is
//! `value_heads * state_size`. An adapted projection adds `<name>.lora_a`
//! and `<name>.lora_b`, and keeps `<name>.weight` for its base.
//!
//! The `linear_attn.*` names are the names `GdnBlock::load` reads. Value
//! heads stay in boostr's tiled order. A Hugging Face Qwen3.5 checkpoint
//! differs, so an export to it needs the edits the `docs/architecture/gdn.md`
//! mapping lists.

use super::layer::{GdnLayer, MIXER};
use super::mixer;
use crate::model::LitGPT;
use crate::prelude::{DType, Result};
use boostr::autograd::{TensorId, Var};

impl<R> GdnLayer<R>
where
    R: boostr::Runtime<DType = DType>,
{
    /// Push every parameter under `prefix` (`layers.N`), in the names the
    /// module docs list, in `trainable_parameters` order.
    ///
    /// # Errors
    ///
    /// The FFN's naming errors.
    pub(in crate::model) fn extend_named<'a>(
        &'a self,
        params: &mut Vec<(String, TensorId, &'a Var<R>)>,
        prefix: &str,
    ) -> Result<()> {
        LitGPT::<R>::push_named_param(
            params,
            format!("{prefix}.input_layernorm.weight"),
            self.input_layernorm.weight(),
        );
        for (name, var) in self.block.named_parameters() {
            let name = mixer::checkpoint_name(&name);
            LitGPT::<R>::push_named_param(params, format!("{prefix}.{MIXER}.{name}"), var);
        }
        LitGPT::<R>::push_named_param(
            params,
            format!("{prefix}.post_attention_layernorm.weight"),
            self.post_attention_layernorm.weight(),
        );
        LitGPT::<R>::extend_ffn_named_trainable(params, prefix, &self.ffn)
    }
}

#[cfg(test)]
mod tests {
    use super::mixer::checkpoint_name;

    #[test]
    fn an_adapted_base_takes_the_plain_weight_name() {
        assert_eq!(
            checkpoint_name("in_proj_qkv.base.weight"),
            "in_proj_qkv.weight"
        );
        assert_eq!(checkpoint_name("in_proj_qkv.lora_a"), "in_proj_qkv.lora_a");
        assert_eq!(checkpoint_name("A_log"), "A_log");
        assert_eq!(checkpoint_name("norm.weight"), "norm.weight");
    }
}
