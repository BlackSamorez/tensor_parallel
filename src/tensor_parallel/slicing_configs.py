"""
Optimized configs for selected models. These configs are not necessary, but they can improve performance in some
cases, e.g. training with very small batches or inference with long sequences.

NB: some of these configs get fairly complicated in order to squeeze a bit of extra performance. When developing your
  own config, you can get most of the performance benefits by using auto config -- and maybe splitting MLP layers.
"""
from functools import partial
from itertools import chain
from typing import Callable, Dict, Sequence

import torch
from transformers import BertConfig, BloomConfig, CodeGenConfig, GPT2Config, GPTNeoXConfig, PretrainedConfig, T5Config

from tensor_parallel.aux_actions import (
    gather_kv,
    select_kv_for_rank,
    split_alibi,
    split_heads,
    split_inner_dim,
    split_num_heads,
)
from tensor_parallel.communications import CollectiveOperation
from tensor_parallel.config import Config
from tensor_parallel.per_device_tensors import PerDeviceTensors
from tensor_parallel.state_actions import Scale, Split, SplitInChunks, SplitInGroupedChunks

ConfigGetter = Callable[[PretrainedConfig, Sequence[torch.device]], Config]


def get_bloom_config(model_config: BloomConfig, devices: Sequence[torch.device]) -> Config:
    world_size = len(devices)
    num_heads = model_config.n_head
    head_dim = model_config.hidden_size // num_heads
    gather_kv_across_ranks = CollectiveOperation(
        world_size=world_size, func=lambda *kvs: gather_kv(*kvs, world_size=world_size)
    )  # this operation ensures that we get attention cache for all heads on each device

    _split_alibi = partial(split_alibi, num_heads=num_heads, world_size=world_size)

    return Config(
        state_rules={
            # BloomAttention
            r".*self_attention\.query_key_value\.(weight|bias)$": SplitInChunks(
                world_size=world_size, dim=0, chunk_size=head_dim * 3
            ),
            r".*self_attention\.dense\.weight$": SplitInChunks(world_size=world_size, dim=1, chunk_size=head_dim),
            r".*self_attention\.dense\.bias$": Scale(world_size=world_size),
            # BloomMLP
            r".*mlp\.dense_h_to_4h\.(weight|bias)$": Split(world_size=world_size, dim=0),
            r".*mlp\.dense_4h_to_h\.weight$": Split(world_size=world_size, dim=1),
            r".*mlp\.dense_4h_to_h\.bias$": Scale(world_size=world_size),
            # BloomModel
            r".*word_embeddings\.weight$": Split(world_size=world_size, dim=1),
            r".*lm_head\.weight$": Split(world_size=world_size, dim=1),
            # note: ^-- lm_head.weight is tied with word_embeddings
        },
        input_rules={
            r".*self_attention$": {"layer_past": select_kv_for_rank, "alibi": _split_alibi},
            r".*lm_head$": {0: "split -1"},  # note: we need to split lm_head inputs because
            # ... lm_head's weights (tied embeddings) are already split across input dimension
        },
        output_rules={
            r".*self_attention$": {1: gather_kv_across_ranks},
            r".*self_attention\.dense$": {0: "sum"},
            r".*mlp\.dense_4h_to_h$": {0: "sum"},
            r".*word_embeddings$": {0: "gather -1"},
            r".*lm_head$": {0: "sum"},
        },
        attr_rules={r".*self_attention$": {"num_heads": partial(split_num_heads, world_size=world_size)}},
    )


def get_t5_config(model_config: T5Config, devices: Sequence[torch.device]) -> Config:
    world_size = len(devices)
    num_heads = model_config.num_heads
    head_dim = model_config.d_kv

    gather_kv_across_ranks = CollectiveOperation(
        world_size=world_size, func=lambda *kvs: gather_kv(*kvs, world_size=world_size)
    )  # this operation ensures that we get attention cache for all heads on each device

    def select_kv_for_rank(*kvs, rank):
        if kvs[0] is None:
            return None
        else:
            if isinstance(kvs[0][0], PerDeviceTensors):
                return (kvs[0][0][rank], kvs[0][1][rank])
            else:
                keys = kvs[0][0]
                values = kvs[0][1]
                return (
                    torch.tensor_split(keys, world_size, dim=1)[rank],
                    torch.tensor_split(values, world_size, dim=1)[rank],
                )

    return Config(
        state_rules={
            # T5Attention
            r".*SelfAttention\.q\.(weight|bias)$": SplitInChunks(world_size=world_size, dim=0, chunk_size=head_dim),
            r".*SelfAttention\.k\.(weight|bias)$": SplitInChunks(world_size=world_size, dim=0, chunk_size=head_dim),
            r".*SelfAttention\.v\.(weight|bias)$": SplitInChunks(world_size=world_size, dim=0, chunk_size=head_dim),
            r".*relative_attention_bias\.weight$": Split(world_size=world_size, dim=1),
            r".*SelfAttention\.o\.weight$": SplitInChunks(world_size=world_size, dim=1, chunk_size=head_dim),
            # T5DenseGatedActDense
            r".*DenseReluDense\.wi\.weight$": Split(world_size=world_size, dim=0),
            r".*DenseReluDense\.wi_0\.weight$": Split(world_size=world_size, dim=0),
            r".*DenseReluDense\.wi_1\.weight$": Split(world_size=world_size, dim=0),
            # T5DenseActDense
            r".*DenseReluDense\.wo\.weight$": Split(world_size=world_size, dim=1),
            # T5Model
            r".*embed_tokens\.weight$": Split(world_size=world_size, dim=1),
            r".*shared\.weight$": Split(world_size=world_size, dim=1),
            r".*lm_head\.weight$": Split(world_size=world_size, dim=1),
            # note: ^-- lm_head.weight tied with word embeddings
        },
        input_rules={
            r".*SelfAttention$": {"past_key_value": select_kv_for_rank},
            r".*lm_head$": {0: "split -1"},  # note: we need to split lm_head inputs because
            # ... lm_head's weights (tied embeddings) are already split across input dimension
        },
        output_rules={
            r".*SelfAttention$": {0: "sum", 1: gather_kv_across_ranks},
            r".*DenseReluDense$": {0: "sum"},
            r".*shared$": {0: "gather -1"},
            r".*embed_tokens$": {0: "gather -1"},
            r".*lm_head$": {0: "sum"},
        },
        attr_rules={
            r".*SelfAttention$": {
                "n_heads": partial(split_num_heads, world_size=world_size),
                "inner_dim": partial(split_inner_dim, num_heads=model_config.num_heads, world_size=world_size),
            },
            r".*relative_attention_bias$": {"embedding_dim": partial(split_num_heads, world_size=world_size)},
        },
    )


def get_bert_config(model_config: BertConfig, devices: Sequence[torch.device]) -> Config:
    world_size = len(devices)
    num_heads = model_config.num_attention_heads
    head_dim = int(model_config.hidden_size / model_config.num_attention_heads)

    return Config(
        state_rules={
            # BertAttention
            r".*self\.query\.(weight|bias)$": SplitInChunks(world_size=world_size, dim=0, chunk_size=head_dim),
            r"self\.key\.(weight|bias)": SplitInChunks(world_size=world_size, dim=0, chunk_size=head_dim),
            r"self\.value\.(weight|bias)": SplitInChunks(world_size=world_size, dim=0, chunk_size=head_dim),
            r".*attention\.output\.dense\.weight$": SplitInChunks(world_size=world_size, dim=1, chunk_size=head_dim),
            r".*attention\.output\.dense\.bias$": Scale(world_size=world_size),
            # BertIntermediate
            r".*intermediate\.dense\.(weight|bias)$": Split(world_size=world_size, dim=0),
            # BertOutput
            r".*[0-9]\.output\.dense\.weight$": Split(world_size=world_size, dim=1),
            r".*[0-9]\.output\.dense\.bias$": Scale(world_size=world_size),
            # BertEmbeddings
            r".*word_embeddings\.weight$": Split(world_size=world_size, dim=1),
            r".*position_embeddings\.weight$": Split(world_size=world_size, dim=1),
            r".*token_type_embeddings\.weight$": Split(world_size=world_size, dim=1),
        },
        input_rules={},
        output_rules={
            r".*attention\.output\.dense$": {0: "sum"},
            r".*[0-9]\.output\.dense$": {0: "sum"},
            r".*word_embeddings$": {0: "gather -1"},
            r".*position_embeddings$": {0: "gather -1"},
            r".*token_type_embeddings$": {0: "gather -1"},
        },
        attr_rules={
            r".*attention\.self$": {
                "num_attention_heads": partial(split_num_heads, world_size=world_size),
                "all_head_size": partial(split_inner_dim, num_heads=num_heads, world_size=world_size),
            },
        },
    )


def get_gpt2_config(model_config: GPT2Config, devices: Sequence[torch.device]) -> Config:
    world_size = len(devices)
    num_heads = model_config.num_attention_heads
    head_dim = model_config.hidden_size // model_config.num_attention_heads

    gather_kv_across_ranks = CollectiveOperation(
        world_size=world_size, func=lambda *kvs: gather_kv(*kvs, world_size=world_size)
    )  # this operation ensures that we get attention cache for all heads on each device

    return Config(
        state_rules={
            # GPT2Attention
            r".*c_attn\.weight$": SplitInGroupedChunks(world_size=world_size, dim=1, num_groups=3, chunk_size=head_dim),
            r".*c_attn\.bias$": SplitInGroupedChunks(world_size=world_size, dim=0, num_groups=3, chunk_size=head_dim),
            r".*q_attn\.weight$": SplitInChunks(world_size=world_size, dim=1, chunk_size=head_dim),
            r".*q_attn\.bias$": SplitInChunks(world_size=world_size, dim=0, chunk_size=head_dim),
            r".*attn\.c_proj\.weight$": SplitInChunks(world_size=world_size, dim=0, chunk_size=head_dim),
            r".*attn\.c_proj\.bias$": Scale(world_size=world_size),
            # GPT2MLP
            r".*c_fc\.weight$": Split(world_size=world_size, dim=1),
            r".*c_fc\.bias$": Split(world_size=world_size, dim=0),
            r".*mlp\.c_proj\.weight$": Split(world_size=world_size, dim=0),
            r".*mlp\.c_proj\.bias$": Scale(world_size=world_size),
            # GPT2Model
            r".*wte\.weight$": Split(world_size=world_size, dim=1),
            r".*wpe\.weight$": Split(world_size=world_size, dim=1),
            r".*lm_head\.weight$": Split(world_size=world_size, dim=1),
            # GPT2LMHeadModel
            # note: ^-- lm_head.weight is tied with word_embeddings
        },
        input_rules={
            r".*[0-9]\.attn$": {"layer_past": select_kv_for_rank},
            r".*lm_head$": {0: "split -1"},  # note: we need to split lm_head inputs because
            # ... lm_head's weights (tied embeddings) are already split across input dimension
        },
        output_rules={
            r".*[0-9]\.attn$": {0: "sum", 1: gather_kv_across_ranks},
            r".*mlp$$": {0: "sum"},
            r".*wte$": {0: "gather -1"},
            r".*wpe$": {0: "gather -1"},
            r".*lm_head$": {0: "sum"},
        },
        attr_rules={
            r".*attn\.c_attn$": {
                "nf": partial(split_inner_dim, num_heads=num_heads, world_size=world_size),
            },
            r".*attn\.q_attn$": {
                "nf": partial(split_inner_dim, num_heads=num_heads, world_size=world_size),
            },
            r".*mlp\.c_fc$": {
                "nf": partial(split_num_heads, world_size=world_size),
            },
            r".*[0-9]\.attn$": {
                "embed_dim": partial(split_inner_dim, num_heads=num_heads, world_size=world_size),
                "num_heads": partial(split_num_heads, world_size=world_size),
                "split_size": partial(split_inner_dim, num_heads=num_heads, world_size=world_size),
            },
        },
    )


def get_gpt_neox_config(model_config: GPTNeoXConfig, devices: Sequence[torch.device]) -> Config:
    world_size = len(devices)
    num_heads = model_config.num_attention_heads
    head_dim = model_config.hidden_size // model_config.num_attention_heads

    gather_kv_across_ranks = CollectiveOperation(
        world_size=world_size, func=lambda *kvs: gather_kv(*kvs, world_size=world_size)
    )  # this operation ensures that we get attention cache for all heads on each device

    return Config(
        state_rules={
            # GPTNeoXAttention
            r".*attention\.query_key_value\.(weight|bias)$": SplitInChunks(
                world_size=world_size, dim=0, chunk_size=head_dim * 3
            ),
            r".*attention\.dense\.weight$": SplitInChunks(world_size=world_size, dim=1, chunk_size=head_dim),
            r".*attention\.dense\.bias$": Scale(world_size=world_size),
            # GPTNeoXMLP
            r".*mlp\.dense_h_to_4h\.(weight|bias)$": Split(world_size=world_size, dim=0),
            r".*mlp\.dense_4h_to_h\.weight$": Split(world_size=world_size, dim=1),
            r".*mlp\.dense_4h_to_h\.bias$": Scale(world_size=world_size),
            # GPTNeoXModel
            r".*embed_in\.weight$": Split(world_size=world_size, dim=1),
            # GPTNeoXForCausalLM
            r".*embed_out\.(weight|bias)$": Split(world_size=world_size, dim=0),
        },
        input_rules={
            r".*attention$": {"layer_past": select_kv_for_rank},
        },
        output_rules={
            r".*attention$": {0: "sum", 1: gather_kv_across_ranks},
            r".*mlp$": {0: "sum"},
            r".*embed_in$": {0: "gather -1"},
            r".*embed_out$": {0: "gather -1"},
        },
        attr_rules={
            r".*attention$": {
                "num_attention_heads": partial(split_num_heads, world_size=world_size),
                "hidden_size": partial(split_inner_dim, num_heads=num_heads, world_size=world_size),
            }
        },
    )


def get_codegen_config(model_config: CodeGenConfig, devices: Sequence[torch.device]) -> Config:
    world_size = len(devices)
    num_heads = model_config.num_attention_heads
    head_dim = model_config.hidden_size // model_config.num_attention_heads

    gather_kv_across_ranks = CollectiveOperation(
        world_size=world_size, func=lambda *kvs: gather_kv(*kvs, world_size=world_size)
    )  # this operation ensures that we get attention cache for all heads on each device

    class SplitCodegenQKV(SplitInChunks):
        def __call__(self, tensor: torch.Tensor, rank: int) -> torch.Tensor:
            tensor = tensor.permute(1, 0)
            tensor = (
                tensor.reshape(tensor.shape[0], 4, 3, -1, head_dim)
                .permute(0, 1, 3, 2, 4)
                .reshape(tensor.shape[0], tensor.shape[1])
            )
            tensor = split_heads(tensor, dim=1, head_dim=12 * head_dim, rank=rank, world_size=world_size)
            result = (
                tensor.reshape(tensor.shape[0], 4, -1, 3, head_dim)
                .permute(0, 1, 3, 2, 4)
                .reshape(tensor.shape[0], tensor.shape[1])
            )
            return result.permute(1, 0)

    def split_codegen_num_heads(num_heads: int, *, rank: int, world_size: int):
        return 4 * split_num_heads(num_heads // 4, rank=rank, world_size=world_size)

    return Config(
        state_rules={
            # CodeGenAttention
            r".*attn\.qkv_proj\.weight$": SplitCodegenQKV(world_size=world_size, chunk_size=head_dim, dim=0),
            r".*attn\.out_proj\.weight$": SplitInChunks(world_size=world_size, dim=1, chunk_size=4 * head_dim),
            # CodeGenMLP
            r".*mlp\.fc_in\.(weight|bias)$": Split(world_size=world_size, dim=0),
            r".*mlp\.fc_out\.weight$": Split(world_size=world_size, dim=1),
            r".*mlp\.fc_out\.bias$": Scale(world_size=world_size),
            # CodeGenModel
            r".*wte\.weight$": Split(world_size=world_size, dim=1),
            # CodeGenForCausalLM
            r".*lm_head\.(weight|bias)$": Split(world_size=world_size, dim=0),
        },
        input_rules={
            r".*attn$": {"layer_past": select_kv_for_rank},
        },
        output_rules={
            r".*attn$": {0: "sum", 1: gather_kv_across_ranks},
            r".*mlp$": {0: "sum"},
            r".*wte$": {0: "gather -1"},
            r".*lm_head$": {0: "gather -1"},
        },
        attr_rules={
            r".*attn$": {
                "embed_dim": partial(split_inner_dim, num_heads=num_heads // 4, world_size=world_size),
                "num_attention_heads": partial(split_codegen_num_heads, world_size=world_size),
            }
        },
    )


def get_llama_config(model_config: PretrainedConfig, devices: Sequence[torch.device]) -> Config:
    # We can't use LlamaConfig since it requires pre-release `transformers``
    assert model_config.model_type == "llama", f"Trying to pass {model_config.model_type} as llama config"

    world_size = len(devices)
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    try:
        num_kv = model_config.num_key_value_heads
        q_per_kv = model_config.num_attention_heads // model_config.num_key_value_heads
        new_modeling = True
    except AttributeError:
        num_kv = model_config.num_attention_heads
        q_per_kv = 1
        new_modeling = False

    gather_kv_across_ranks = CollectiveOperation(
        world_size=world_size, func=lambda *kvs: gather_kv(*kvs, world_size=world_size)
    )  # this operation ensures that we get attention cache for all heads on each device

    config = Config(
        state_rules={
            # LlamaAttention
            r".*self_attn\.q_proj\.weight$": SplitInChunks(
                world_size=world_size, dim=0, chunk_size=q_per_kv * head_dim
            ),
            r".*self_attn\.k_proj\.weight$": SplitInChunks(world_size=world_size, dim=0, chunk_size=head_dim),
            r".*self_attn\.v_proj\.weight$": SplitInChunks(world_size=world_size, dim=0, chunk_size=head_dim),
            r".*self_attn\.o_proj\.weight$": SplitInChunks(
                world_size=world_size, dim=1, chunk_size=q_per_kv * head_dim
            ),
            # LlamaFeedForward
            r".*mlp\.gate_proj\.weight$": Split(world_size=world_size, dim=0),
            r".*mlp\.down_proj\.weight$": Split(world_size=world_size, dim=1),
            r".*mlp\.up_proj\.weight$": Split(world_size=world_size, dim=0),
            # LlamaModel
            r".*embed_tokens.weight$": Split(world_size=world_size, dim=1),
            r".*lm_head\.weight$": Split(world_size=world_size, dim=0),
        },
        input_rules={
            r".*self_attn$": {"past_key_value": select_kv_for_rank},
        },
        output_rules={
            r".*self_attn$": {0: "sum", 2: gather_kv_across_ranks},
            r".*mlp$": {0: "sum"},
            r".*embed_tokens$": {0: "gather -1"},
            r".*lm_head$": {0: "gather -1"},
        },
        attr_rules={
            r".*self_attn$": {
                "hidden_size": partial(split_inner_dim, num_heads=num_kv, world_size=world_size),
                "num_heads": lambda n, rank: q_per_kv
                * split_num_heads(n // q_per_kv, rank=rank, world_size=world_size),
            }
        },
    )

    if new_modeling:
        config.attr_rules[r".*self_attn$"]["num_key_value_heads"] = partial(split_num_heads, world_size=world_size)

    return config


def get_refined_web_config(model_config: PretrainedConfig, devices: Sequence[torch.device]) -> Config:
    # We can't use `RWConfig`` since it's custom code
    assert model_config.model_type == "RefinedWeb", f"Trying to pass {model_config.model_type} as RefinedWeb config"
    assert not model_config.bias and not model_config.alibi, f"Running Falcon with biases or alibi is not supported"

    world_size = len(devices)
    hidden_size = model_config.hidden_size
    num_heads = model_config.n_head
    num_kv = model_config.n_head_kv
    head_dim = hidden_size // num_heads
    q_per_kv = num_heads // num_kv

    head_dim = model_config.hidden_size // model_config.num_attention_heads

    gather_kv_across_ranks = CollectiveOperation(
        world_size=world_size, func=lambda *kvs: gather_kv(*kvs, world_size=world_size)
    )  # this operation ensures that we get attention cache for all heads on each device

    return Config(
        state_rules={
            # Attention
            r".*self_attention\.query_key_value\.weight$": SplitInChunks(
                world_size=world_size, dim=0, chunk_size=(2 + q_per_kv) * head_dim
            ),
            r".*self_attention\.dense\.weight$": SplitInChunks(
                world_size=world_size, dim=1, chunk_size=q_per_kv * head_dim
            ),
            # MLP
            r".*mlp\.dense_h_to_4h\.weight$": Split(world_size=world_size, dim=0),
            r".*mlp\.dense_4h_to_h\.weight$": Split(world_size=world_size, dim=1),
            # RWModel
            r".*word_embeddings\.weight$": Split(world_size=world_size, dim=1),
            # RWForCausalLM
            r".*lm_head\.weight$": Split(world_size=world_size, dim=1),
        },
        input_rules={
            r".*self_attention$": {"layer_past": select_kv_for_rank},
            r".*lm_head$": {0: "split -1"},  # note: we need to split lm_head inputs because
            # ... lm_head's weights (tied embeddings) are already split across input dimension
        },
        output_rules={
            r".*self_attention$": {0: "sum", 1: gather_kv_across_ranks},
            r".*\.mlp$": {0: "sum"},
            r".*word_embeddings$": {0: "gather -1"},
            r".*lm_head$": {0: "sum"},
        },
        attr_rules={
            r".*self_attention$": {
                "num_kv": partial(split_num_heads, world_size=world_size),
                "num_heads": lambda n, rank: q_per_kv
                * split_num_heads(n // q_per_kv, rank=rank, world_size=world_size),
            }
        },
    )


PREDEFINED_CONFIGS: Dict[str, ConfigGetter] = {
    "bloom": get_bloom_config,
    "t5": get_t5_config,
    "bert": get_bert_config,
    "gpt2": get_gpt2_config,
    "gpt_neox": get_gpt_neox_config,
    "codegen": get_codegen_config,
    "llama": get_llama_config,
    "RefinedWeb": get_refined_web_config,
}
