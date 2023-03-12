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
from transformers import BertConfig, BloomConfig, GPT2Config, GPTNeoXConfig, PretrainedConfig, T5Config

from tensor_parallel.communications import CollectiveOperation
from tensor_parallel.slicer_wrapper import Config
from tensor_parallel.tensor_parallel import PerDeviceTensors
from tensor_parallel.utils import nested_map

ConfigGetter = Callable[[PretrainedConfig, Sequence[torch.device]], Config]


def gather_kv(*present_key_value_state, world_size):
    if present_key_value_state[0] is None:
        return present_key_value_state
    else:
        return [tuple(PerDeviceTensors(*item) for item in zip(*present_key_value_state))] * world_size


def select_kv_for_rank(present_key_value_state, rank):
    return nested_map(lambda x: x[rank] if isinstance(x, PerDeviceTensors) else x, present_key_value_state)


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
            r".*self_attention\.query_key_value\.(weight|bias)$": (
                partial(split_heads, dim=0, head_dim=head_dim * 3, world_size=world_size),
                "split 0",
            ),
            r".*self_attention\.dense\.weight$": (
                partial(split_heads, dim=1, head_dim=head_dim, world_size=world_size),
                "split 1",
            ),
            r".*self_attention\.dense\.bias$": "scale",
            # BloomMLP
            r".*mlp\.dense_h_to_4h\.(weight|bias)$": "split 0",
            r".*mlp\.dense_4h_to_h\.weight$": "split 1",
            r".*mlp\.dense_4h_to_h\.bias$": "scale",
            # BloomModel
            r".*word_embeddings.weight$": "split 1",
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


def split_heads(tensor: torch.Tensor, *, dim: int, head_dim: int, rank: int, world_size: int, optional: bool = False):
    """Split a tensor along dim such that each part size is divisible by head_dim"""
    if tensor is None and optional:
        return None
    assert tensor.shape[dim] % head_dim == 0, tensor.shape
    if dim < 0:
        dim = (tensor.ndim + dim) % tensor.ndim
    shape = list(tensor.shape)
    shape[dim] //= head_dim
    shape.insert(dim + 1, head_dim)
    tensor_part = tensor.reshape(shape).tensor_split(world_size, dim=dim)[rank].flatten(dim, dim + 1)
    if tensor_part.shape[dim] == 0:
        shape = list(tensor_part.shape)
        shape[dim] = 1
        return torch.zeros(shape)
    return tensor_part


def split_num_heads(num_heads: int, *, rank: int, world_size: int):
    assigned_num_heads = torch.empty(num_heads, device="meta").tensor_split(world_size)[rank].numel()
    return assigned_num_heads if assigned_num_heads != 0 else 1


def split_inner_dim(inner_dim: int, *, rank: int, num_heads: int, world_size: int):
    return split_num_heads(num_heads=num_heads, rank=rank, world_size=world_size) * (inner_dim // num_heads)


def split_alibi(alibi: torch.Tensor, *, rank: int, num_heads: int, world_size: int) -> torch.Tensor:
    """split alibi tensor of shape [batch_size * num_heads, ...] over attention heads"""
    alibi_expanded = alibi.reshape(-1, num_heads, *alibi.shape[1:])
    alibi_part = alibi_expanded.tensor_split(world_size, dim=1)[rank]
    return alibi_part.reshape(-1, *alibi.shape[1:])


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
            r".*SelfAttention\.q\.(weight|bias)$": (
                partial(split_heads, dim=0, head_dim=head_dim, world_size=world_size),
                "split 0",
            ),
            r".*SelfAttention\.k\.(weight|bias)$": (
                partial(split_heads, dim=0, head_dim=head_dim, world_size=world_size),
                "split 0",
            ),
            r".*SelfAttention\.v\.(weight|bias)$": (
                partial(split_heads, dim=0, head_dim=head_dim, world_size=world_size),
                "split 0",
            ),
            r".*relative_attention_bias\.weight$": "split 1",
            r".*SelfAttention\.o\.weight$": (
                partial(split_heads, dim=1, head_dim=head_dim, world_size=world_size),
                "split 1",
            ),
            # T5DenseGatedActDense
            r".*DenseReluDense\.wi\.weight$": "split 0",
            r".*DenseReluDense\.wi_0\.weight$": "split 0",
            r".*DenseReluDense\.wi_1\.weight$": "split 0",
            # T5DenseActDense
            r".*DenseReluDense\.wo\.weight$": "split 1",
            # T5Model
            r".*shared.weight$": "split 1",
            r".*lm_head\.weight$": "split 1",
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
            r".*self\.query\.(weight|bias)$": (
                partial(split_heads, dim=0, head_dim=head_dim, world_size=world_size),
                "split 0",
            ),
            r"self\.key\.(weight|bias)": (
                partial(split_heads, dim=0, head_dim=head_dim, world_size=world_size),
                "split 0",
            ),
            r"self\.value\.(weight|bias)": (
                partial(split_heads, dim=0, head_dim=head_dim, world_size=world_size),
                "split 0",
            ),
            r".*attention\.output\.dense\.weight$": (
                partial(split_heads, dim=1, head_dim=head_dim, world_size=world_size),
                "split 1",
            ),
            r".*attention\.output\.dense\.bias$": "scale",
            # BertIntermediate
            r".*intermediate\.dense\.(weight|bias)$": "split 0",
            # BertOutput
            r".*[0-9]\.output\.dense\.weight$": "split 1",
            r".*[0-9]\.output\.dense\.bias$": "scale",
            # BertEmbeddings
            r".*word_embeddings\.weight$": "split 1",
            r".*position_embeddings\.weight$": "split 1",
            r".*token_type_embeddings\.weight$": "split 1",
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

    def split_gpt2_qkv(tensor: torch.Tensor, rank: int, dim: int, world_size: int, head_dim: int, num_parts: int):
        assert tensor.shape[dim] % num_parts == 0
        dims = list(tensor.shape)
        dims.insert(-1, num_parts)
        dims[-1] //= num_parts

        some_tensor = tensor.view(*dims)
        new_tensor = torch.cat(
            [
                split_heads(some_tensor[..., i, :], dim=-1, head_dim=head_dim, rank=rank, world_size=world_size)
                for i in range(num_parts)
            ],
            dim=-1,
        )
        return new_tensor

    return Config(
        state_rules={
            # GPT2Attention
            r".*c_attn\.weight$": (
                partial(split_gpt2_qkv, dim=1, head_dim=head_dim, num_parts=3, world_size=world_size),
                "split 1",
            ),
            r".*c_attn\.bias$": (
                partial(split_gpt2_qkv, dim=0, head_dim=head_dim, num_parts=3, world_size=world_size),
                "split 0",
            ),
            r".*q_attn\.weight$": (partial(split_heads, dim=1, head_dim=head_dim, world_size=world_size), "split 1"),
            r".*q_attn\.bias$": (partial(split_heads, dim=0, head_dim=head_dim, world_size=world_size), "split 0"),
            r".*attn\.c_proj\.weight$": (
                partial(split_heads, dim=0, head_dim=head_dim, world_size=world_size),
                "split 0",
            ),
            r".*attn\.c_proj\.bias$": "scale",
            # GPT2MLP
            r".*c_fc\.weight$": "split 1",
            r".*c_fc\.bias$": "split 0",
            r".*mlp\.c_proj\.weight$": "split 0",
            r".*mlp\.c_proj\.bias$": "scale",
            # GPT2Model
            r".*wte\.weight$": "split 1",
            r".*wpe\.weight$": "split 1",
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
            r".*attention\.query_key_value\.(weight|bias)$": (
                partial(split_heads, dim=0, head_dim=head_dim * 3, world_size=world_size),
                "split 0",
            ),
            r".*attention\.dense\.weight$": (
                partial(split_heads, dim=1, head_dim=head_dim, world_size=world_size),
                "split 1",
            ),
            r".*attention\.dense\.bias$": "scale",
            # GPTNeoXMLP
            r".*mlp\.dense_h_to_4h\.(weight|bias)$": "split 0",
            r".*mlp\.dense_4h_to_h\.weight$": "split 1",
            r".*mlp\.dense_4h_to_h\.bias$": "scale",
            # GPTNeoXModel
            r".*embed_in\.weight$": "split 1",
            # GPTNeoXForCausalLM
            r".*embed_out\.(weight|bias)$": "split 0",
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


PREDEFINED_CONFIGS: Dict[str, ConfigGetter] = {
    "bloom": get_bloom_config,
    "t5": get_t5_config,
    "bert": get_bert_config,
    "gpt2": get_gpt2_config,
    "gpt_neox": get_gpt_neox_config,
}
