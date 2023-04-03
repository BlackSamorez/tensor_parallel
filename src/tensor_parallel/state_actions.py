"""slicing_actions.py: file contains actions """

from abc import ABC, abstractclassmethod
from typing import Any, Callable, Optional, Sequence

import torch
from torch import Tensor


class StateAction(ABC):
    @abstractclassmethod
    def __call__(self, tensor: Tensor, rank: int) -> Tensor:
        pass

    @abstractclassmethod
    def undo(self, tensors: Sequence[Tensor]) -> Tensor:
        pass


class LegacyStateAction(StateAction):
    def __init__(
        self,
        action: Callable[[Tensor, int], Tensor],
        reverse_action: Optional[Callable[[Sequence[Tensor]], Tensor]] = None,
    ):
        self.action = action
        self.reverse_action = reverse_action

    def __call__(self, tensor: Tensor, rank: int) -> Tensor:
        return self.action(tensor, rank)

    def undo(self, tensors: Sequence[Tensor]) -> Tensor:
        if self.reverse_action is None:
            raise Exception(f"No reverse action provided for {self.action}. Can't undo.")

        return self.reverse_action(tensors)


class Split(StateAction):
    """XXXXXXXXXXXXXXX -> (world_size = 3) -> [XXXXX, XXXXX, XXXXX]"""

    def __init__(self, world_size: int, dim: int):
        super().__init__()
        self.world_size = world_size
        self.dim = dim

    def __call__(self, tensor: Tensor, rank: int) -> Tensor:
        return torch.tensor_split(tensor, self.world_size, dim=self.dim)[rank]

    def undo(self, tensors: Sequence[Tensor]) -> Tensor:
        return torch.cat([tensor.cpu() for tensor in tensors], dim=self.dim)


class Scale(StateAction):
    def __init__(self, world_size: int):
        super().__init__()
        self.world_size = world_size

    def __call__(self, tensor: Tensor, rank: int) -> Tensor:
        return tensor / self.world_size

    def undo(self, tensors: Sequence[Tensor]) -> Tensor:
        return tensors[0] * self.world_size


class SplitInChunks(StateAction):
    """AAABBBCCCDDDEEE -> (world_size = 3, chunk_size = 3) -> [AAABBB, CCCDDD, EEE]
    Split retaining whole chunks
    """

    def __init__(self, world_size: int, dim: int, chunk_size: int, optional: bool = False):
        super().__init__()
        self.world_size = world_size
        self.dim = dim
        self.chunk_size = chunk_size
        self.optional = optional

    def __call__(self, tensor: Tensor, rank: int) -> Tensor:
        if tensor is None and self.optional:
            return None
        dim = self.dim
        assert tensor.shape[dim] % self.chunk_size == 0, tensor.shape
        if dim < 0:
            dim = (tensor.ndim + dim) % tensor.ndim
        shape = list(tensor.shape)
        shape[dim] //= self.chunk_size
        shape.insert(dim + 1, self.chunk_size)
        tensor_part = tensor.reshape(shape).tensor_split(self.world_size, dim=dim)[rank].flatten(dim, dim + 1)
        return tensor_part

    def undo(self, tensors: Sequence[Tensor]) -> Tensor:
        return torch.cat([tensor.cpu() for tensor in tensors], dim=self.dim)


class SplitInsideChunks(StateAction):
    """AAABBBCCCDDDEEE -> (world_size = 3, num_chunks = 5) -> [ABCDE, ABCDE, ABCDE]"""

    def __init__(self, world_size: int, dim: int, num_chunks: int) -> None:
        super().__init__()
        self.world_size = world_size
        self.dim = dim
        self.num_chunks = num_chunks

    def __call__(self, tensor: Tensor, rank: int) -> Tensor:
        shape = list(tensor.shape)
        shape[self.dim] = shape[self.dim] // self.num_chunks
        shape.insert(self.dim, self.num_chunks)
        grouped_tensor = tensor.reshape(*shape)
        grouped_shard = torch.tensor_split(grouped_tensor, self.world_size, dim=self.dim + 1)[rank]
        return torch.flatten(grouped_shard, start_dim=self.dim, end_dim=self.dim + 1)

    def undo(self, tensors: Sequence[Tensor]) -> Tensor:
        grouped_tensor = []
        for tensor in tensors:
            shape = list(input.shape)
            shape[self.dim] = shape[self.dim] // self.num_chunks
            shape.insert(self.dim, self.num_chunks)
            grouped_tensor.append(tensor.reshape(*shape).cpu())

        output_shape = tensors[0].shape
        del output_shape[self.dim]
        output_shape[self.dim] = -1

        return torch.cat(grouped_tensor, dim=self.dim).reshape(*output_shape)
