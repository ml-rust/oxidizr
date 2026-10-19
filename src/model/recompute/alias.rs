//! `alias` for every trainable block type: a cheap duplicate that keeps each
//! parameter's `TensorId` and trainability.
//!
//! A checkpointed layer captures its alias by value in a `'static` closure.
//! Each `Var` goes through `Var::alias` or a boostr `alias`, never `Clone`:
//! a `clone` mints fresh ids and orphans the gradients.

use crate::model::{
    Block, BlockType, CausalSelfAttention, FfnLayer, Mlp, MoeLayer, MultiHeadLatentAttention,
    RotaryEmbedding,
};
use crate::prelude::DType;

impl<R> RotaryEmbedding<R>
where
    R: boostr::Runtime<DType = DType>,
{
    /// The same cos and sin tables under the same ids.
    pub(in crate::model) fn alias(&self) -> Self {
        Self {
            cos: self.cos.alias(),
            sin: self.sin.alias(),
            _dim: self._dim,
        }
    }
}

impl<R> MultiHeadLatentAttention<R>
where
    R: boostr::Runtime<DType = DType>,
{
    /// The same MLA projections and norms under the same ids.
    pub(in crate::model) fn alias(&self) -> Self {
        Self {
            mla: self.mla.alias(),
        }
    }
}

impl<R> CausalSelfAttention<R>
where
    R: boostr::Runtime<DType = DType>,
{
    /// The same projections, Q/K norms and RoPE tables under the same ids.
    pub(in crate::model) fn alias(&self) -> Self {
        Self {
            q_proj: self.q_proj.alias(),
            k_proj: self.k_proj.alias(),
            v_proj: self.v_proj.alias(),
            o_proj: self.o_proj.alias(),
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            q_norm: self.q_norm.as_ref().map(boostr::nn::RmsNorm::alias),
            k_norm: self.k_norm.as_ref().map(boostr::nn::RmsNorm::alias),
            rope: self.rope.alias(),
            sliding_window: self.sliding_window,
        }
    }
}

impl<R> Mlp<R>
where
    R: boostr::Runtime<DType = DType>,
{
    /// The same three projections under the same ids.
    pub(in crate::model) fn alias(&self) -> Self {
        Self {
            gate_proj: self.gate_proj.alias(),
            up_proj: self.up_proj.alias(),
            down_proj: self.down_proj.alias(),
        }
    }
}

impl<R> MoeLayer<R>
where
    R: boostr::Runtime<DType = DType>,
{
    /// The same router gate, routed experts and shared expert under the same ids.
    pub(in crate::model) fn alias(&self) -> Self {
        Self {
            moe: self.moe.alias(),
        }
    }
}

impl<R> FfnLayer<R>
where
    R: boostr::Runtime<DType = DType>,
{
    /// The same MLP or MoE under the same ids.
    pub(in crate::model) fn alias(&self) -> Self {
        match self {
            FfnLayer::Mlp(mlp) => FfnLayer::Mlp(Box::new(mlp.alias())),
            FfnLayer::Moe(moe) => FfnLayer::Moe(Box::new(moe.alias())),
        }
    }
}

impl<R> BlockType<R>
where
    R: boostr::Runtime<DType = DType>,
{
    /// The same arm over the same parameters under the same ids.
    pub(in crate::model) fn alias(&self) -> Self {
        match self {
            BlockType::Mamba { block, norm } => BlockType::Mamba {
                block: Box::new(block.alias()),
                norm: norm.alias(),
            },
            BlockType::Mamba2 { block, norm } => BlockType::Mamba2 {
                block: Box::new(block.alias()),
                norm: norm.alias(),
            },
            BlockType::Mamba3 { block, norm } => BlockType::Mamba3 {
                block: Box::new(block.alias()),
                norm: norm.alias(),
            },
            BlockType::Attention {
                attn,
                ffn,
                input_layernorm,
                post_attention_layernorm,
            } => BlockType::Attention {
                attn: Box::new(attn.alias()),
                ffn: ffn.alias(),
                input_layernorm: input_layernorm.alias(),
                post_attention_layernorm: post_attention_layernorm.alias(),
            },
            BlockType::LegacyAttention {
                attn,
                ffn,
                input_layernorm,
                post_attention_layernorm,
            } => BlockType::LegacyAttention {
                attn: Box::new(attn.alias()),
                ffn: ffn.alias(),
                input_layernorm: input_layernorm.alias(),
                post_attention_layernorm: post_attention_layernorm.alias(),
            },
            BlockType::Gdn(layer) => BlockType::Gdn(Box::new(layer.alias())),
        }
    }
}

impl<R> Block<R>
where
    R: boostr::Runtime<DType = DType>,
{
    /// The same layer over the same parameters under the same ids.
    pub(in crate::model) fn alias(&self) -> Self {
        Self {
            block_type: self.block_type.alias(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::{self, Model};
    use crate::model::{Block, BlockType, FfnLayer};
    use boostr::autograd::{TensorId, Var};
    use boostr::CpuRuntime;

    type Ids = Vec<(TensorId, bool)>;

    fn ids(params: Vec<(TensorId, &Var<CpuRuntime>)>) -> Ids {
        params
            .into_iter()
            .map(|(id, var)| (id, var.requires_grad()))
            .collect()
    }

    fn block_ids(block: &Block<CpuRuntime>) -> Ids {
        ids(block.trainable_parameters())
    }

    /// Every layer of `model` and its alias list the same ids and flags.
    fn assert_layers_alias(model: &Model, label: &str) {
        for (i, layer) in model.layers.iter().enumerate() {
            let want = block_ids(layer);
            assert!(
                !want.is_empty(),
                "{label} layer {i}: no trainable parameter"
            );
            assert_eq!(block_ids(&layer.alias()), want, "{label} layer {i}");
        }
    }

    fn ffn_ids(ffn: &FfnLayer<CpuRuntime>) -> Ids {
        ids(ffn.trainable_parameters())
    }

    #[test]
    fn mamba_blocks_keep_their_ids() {
        assert_layers_alias(&fixtures::build(&fixtures::mamba1()), "mamba1");
        assert_layers_alias(&fixtures::build(&fixtures::mamba2()), "mamba2");
        assert_layers_alias(&fixtures::build(&fixtures::mamba3()), "mamba3");
    }

    #[test]
    fn mla_blocks_keep_their_ids() {
        for (cfg, label) in [
            (fixtures::mla_mlp(), "mla+mlp"),
            (fixtures::mla_moe(), "mla+moe"),
        ] {
            let model = fixtures::build(&cfg);
            assert_layers_alias(&model, label);
            let BlockType::Attention { attn, ffn, .. } = &model.layers[0].block_type else {
                panic!("{label}: layer 0 must be an MLA block");
            };
            assert_eq!(
                ids(attn.alias().trainable_parameters()),
                ids(attn.trainable_parameters())
            );
            assert_eq!(ffn_ids(&ffn.alias()), ffn_ids(ffn));
        }
    }

    #[test]
    fn legacy_blocks_keep_their_ids() {
        for (cfg, label) in [
            (fixtures::legacy_mlp(), "legacy+mlp"),
            (fixtures::legacy_moe(), "legacy+moe"),
        ] {
            let model = fixtures::build(&cfg);
            assert_layers_alias(&model, label);
            let BlockType::LegacyAttention { attn, ffn, .. } = &model.layers[0].block_type else {
                panic!("{label}: layer 0 must be a legacy attention block");
            };
            let alias = attn.alias();
            assert_eq!(
                ids(alias.trainable_parameters()),
                ids(attn.trainable_parameters())
            );
            assert_eq!(alias.rope.cos.id(), attn.rope.cos.id());
            assert_eq!(alias.rope.sin.id(), attn.rope.sin.id());
            assert_eq!(ffn_ids(&ffn.alias()), ffn_ids(ffn));
        }
    }

    #[test]
    fn gdn_blocks_keep_their_ids() {
        assert_layers_alias(&fixtures::build(&fixtures::gdn_moe()), "gdn+moe");
    }

    /// A frozen LoRA base keeps its id and stays frozen in the alias.
    #[test]
    fn a_frozen_lora_base_keeps_its_id_and_flag() {
        let model = fixtures::build(&fixtures::legacy_lora_frozen());
        assert_layers_alias(&model, "legacy+lora");
        let BlockType::LegacyAttention { attn, .. } = &model.layers[0].block_type else {
            panic!("layer 0 must be a legacy attention block");
        };
        let alias = attn.alias();
        let flag = |var: Option<&Var<CpuRuntime>>| var.map(|v| (v.id(), v.requires_grad()));
        let base = flag(attn.q_proj.weight());
        assert_eq!(base.map(|(_, trains)| trains), Some(false));
        assert_eq!(flag(alias.q_proj.weight()), base);
    }
}
