"""slicing_actions.py: file contains actions """

from abc import ABC, abstractclassmethod
from typing import Any

import torch
from torch import Tensor

from tensor_parallel.communications import CollectiveOperation
from tensor_parallel.per_device_tensors import PerDeviceTensors


class Action(ABC):
    @abstractclassmethod
    def __call__(self, input: Any, rank: int) -> Any:
        pass


class Split(Action):
    """XXXXXXXXXXXXXXX -> (world_size = 3) -> [XXXXX, XXXXX, XXXXX]"""

    def __init__(self, world_size: int, dim: int):
        super().__init__()
        self.world_size = world_size
        self.dim = dim

    def __call__(self, input: Tensor, rank: int) -> Tensor:
        return torch.tensor_split(input, self.world_size, dim=self.dim)[rank]


class Scale(Action):
    def __init__(self, world_size: int):
        super().__init__()
        self.world_size = world_size

    def __call__(self, input: Tensor, rank: int) -> Tensor:
        return input / self.world_size


class SplitInChunks(Action):
    """AAABBBCCCDDDEEE -> (world_size = 3, chunk_size = 3) -> [AAABBB, CCCDDD, EEE]"""

    def __init__(self, world_size: int, dim: int, chunk_size: int, optional: bool = False):
        super().__init__()
        self.world_size = world_size
        self.dim = dim
        self.chunk_size = chunk_size
        self.optional = optional

    def __call__(self, input: Tensor, rank: int) -> Tensor:
        if input is None and self.optional:
            return None
        dim = self.dim
        assert input.shape[dim] % self.chunk_size == 0, input.shape
        if dim < 0:
            dim = (input.ndim + dim) % input.ndim
        shape = list(input.shape)
        shape[dim] //= self.chunk_size
        shape.insert(dim + 1, self.chunk_size)
        tensor_part = input.reshape(shape).tensor_split(self.world_size, dim=dim)[rank].flatten(dim, dim + 1)
        return tensor_part


class SplitNumChunks(Action):
    """5 -> (world_size = 3) -> [2, 2 ,1]"""

    def __init__(self, world_size: int):
        super().__init__()
        self.world_size = world_size

    def __call__(self, num_chunks: int, rank: int) -> int:
        return torch.empty(num_chunks, device="meta").tensor_split(self.world_size)[rank].numel()


class SplitDimInChunks(Action):
    """15 -> (world_size = 3, chunk_size = 3) -> [6, 6, 3]"""

    def __init__(self, world_size: int, num_chunks: int):
        super().__init__()
        self.world_size = world_size
        self.num_chunks = num_chunks

    def __call__(self, inner_dim: int, rank: int) -> int:
        return torch.empty(self.num_chunks, device="meta").tensor_split(self.world_size)[rank].numel() * (
            inner_dim // self.num_chunks
        )


class SplitInsideChunks(Action):
    """AAABBBCCCDDDEEE -> (world_size = 3, num_chunks = 5) -> [ABCDE, ABCDE, ABCDE]"""

    def __init__(self, world_size: int, dim: int, num_chunks: int) -> None:
        super().__init__()
        self.world_size = world_size
        self.dim = dim
        self.num_chunks = num_chunks

    def __call__(self, input: Tensor, rank: int) -> Tensor:
        shape = list(input.shape)
        shape[self.dim] = shape[self.dim] // self.num_chunks
        shape.insert(self.dim, self.num_chunks)
        grouped_tensor = input.reshape(*shape)
        grouped_shard = torch.tensor_split(grouped_tensor, self.world_size, dim=self.dim + 1)[rank]
        return torch.flatten(grouped_shard, start_dim=self.dim, end_dim=self.dim + 1)


class GatherKV(Action):
    def __init__(self, world_size: int):
        super().__init__()
        self.world_size = world_size

    def __call__(self, present_key_value_state, rank: int):
        if present_key_value_state[0] is None:
            return present_key_value_state
        else:
            return [tuple(PerDeviceTensors(*item) for item in present_key_value_state)] * self.world_size


class SplitAlibi(Action):
    def __init__(self, world_size: int, num_heads: int):
        super().__init__()
        self.world_size = world_size
        self.num_heads = num_heads

    def __call__(self, alibi: Tensor, rank: int) -> Tensor:
        alibi_expanded = alibi.reshape(-1, self.num_heads, *alibi.shape[1:])
        alibi_part = alibi_expanded.tensor_split(self.world_size, dim=1)[rank]
        return alibi_part.reshape(-1, *alibi.shape[1:])


def gather_kv(world_size: int) -> Action:
    """Constructs an Action for gathering attention caches across ranks"""

    def operation(*present_key_value_state):
        if present_key_value_state[0] is None:
            return present_key_value_state
        else:
            return [tuple(PerDeviceTensors(*item) for item in zip(*present_key_value_state))] * world_size

    return CollectiveOperation(world_size=world_size, func=operation)
