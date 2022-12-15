"""
Optimized configs for selected models. These configs are not necessary, but they can improve performance in some
cases, e.g. training with very small batches or inference with long sequences.

NB: some of these configs get fairly complicated in order to squeeze a bit of extra performance. When developing your
  own config, you can get most of the performance benefits by using auto config -- and maybe splitting MLP layers.
"""

from transformers import BloomConfig


def split_heads(tensor: torch.Tensor, *, dim: int, head_dim: int, rank: int, world_size: int):
    """Split a tensor along dim such that each part size is divisible by head_dim"""
    if dim < 0:
        dim = (tensor.ndim + dim) % tensor.ndim
    shape = list(tensor.shape)
    shape[dim] //= head_dim
    shape.insert(dim + 1, head_dim)
    tensor_part = tensor.reshape(shape).tensor_split(world_size, dim=dim)[rank].flatten(dim, dim + 1)
    return tensor_part


def get_bloom_config(bloom_config: BloomConfig, devices: Sequence[torch.device]):
    return Config(
        state_rules={
            ".*self_attention\.query_key_value\.(weight|bias)": "split 0",
            ".*self_attention\.dense\.(weight|bias)": "split 0",
            ".*mlp\.dense_h_to_4h\.(weight|bias)": "split 0",
            ".*mlp\.dense_4h_to_h\.weight": "split 1",
            ".*mlp\.dense_4h_to_h\.bias": "scale",
        },
        input_rules={".*self_attention\.query_key_value": {"layer_past": func}},
        output_rules={
            ".*self_attention\.query_key_value": {0: "gather -1"},
            ".*self_attention\.dense": {0: "gather -1"},
            ".*mlp\.dense_4h_to_h$": {0: "sum"},
        },
        attr_rules={},
    )
