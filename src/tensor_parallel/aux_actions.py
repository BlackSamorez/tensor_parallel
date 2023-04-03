import torch

from tensor_parallel.per_device_tensors import PerDeviceTensors
from tensor_parallel.utils import nested_map


def gather_kv(*present_key_value_state, world_size):
    if present_key_value_state[0] is None:
        return present_key_value_state
    else:
        return [tuple(PerDeviceTensors(*item) for item in zip(*present_key_value_state))] * world_size


def select_kv_for_rank(present_key_value_state, rank):
    return nested_map(lambda x: x[rank] if isinstance(x, PerDeviceTensors) else x, present_key_value_state)


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
        return torch.zeros(shape[:dim] + shape[dim + 1 :])
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
