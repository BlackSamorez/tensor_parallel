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
from transformers import BloomConfig, PretrainedConfig

from tensor_parallel.communications import CollectiveOperation
from tensor_parallel.slicer_wrapper import Config
from tensor_parallel.tensor_parallel import PerDeviceTensors

ConfigGetter = Callable[[PretrainedConfig, Sequence[torch.device]], Config]


def get_bloom_config(model_config: BloomConfig, devices: Sequence[torch.device]) -> Config:
    world_size = len(devices)
    num_heads = model_config.n_head
    head_dim = model_config.hidden_size // num_heads
    gather_kv_across_ranks = CollectiveOperation(
        world_size=world_size, func=lambda *kvs: [PerDeviceTensors(*chain(*kvs))] * world_size
    )  # this operation ensures that we get attention cache for all heads on each device

    select_kv_for_rank = lambda kvs, rank: (kvs[2 * rank], kvs[2 * rank + 1]) if kvs else None

    _split_alibi = partial(split_alibi, num_heads=num_heads, world_size=world_size)

    return Config(
        state_rules={
            r".*self_attention\.query_key_value\.(weight|bias)$": partial(
                split_heads, dim=0, head_dim=head_dim * 3, world_size=world_size
            ),
            r".*self_attention\.dense\.weight$": partial(split_heads, dim=1, head_dim=head_dim, world_size=world_size),
            r".*self_attention\.dense\.bias$": "scale",
            r".*mlp\.dense_h_to_4h\.(weight|bias)$": "split 0",
            r".*mlp\.dense_4h_to_h\.weight$": "split 1",
            r".*mlp\.dense_4h_to_h\.bias$": "scale",
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
    return tensor_part


def split_num_heads(num_heads: int, *, rank: int, world_size: int):
    return torch.empty(num_heads, device="meta").tensor_split(world_size)[rank].numel()


def split_alibi(alibi: torch.Tensor, *, rank: int, num_heads: int, world_size: int) -> torch.Tensor:
    """split alibi tensor of shape [batch_size * num_heads, ...] over attention heads"""
    alibi_expanded = alibi.reshape(-1, num_heads, *alibi.shape[1:])
    alibi_part = alibi_expanded.tensor_split(world_size, dim=1)[rank]
    return alibi_part.reshape(-1, *alibi.shape[1:])


PREDEFINED_CONFIGS: Dict[str, ConfigGetter] = {
    "BloomModel": get_bloom_config,
}
