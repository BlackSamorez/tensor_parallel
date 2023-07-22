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


class SplitInChunks(Split):
    """AAABBBCCCDDDEEE -> (world_size = 3, chunk_size = 3) -> [AAABBB, CCCDDD, EEE]
    Split retaining whole chunks
    """

    def __init__(self, world_size: int, dim: int, chunk_size: int, optional: bool = False):
        super().__init__(world_size, dim)
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


class SplitInGroupedChunks(Split):
    """AABBCCDDEE AABBCCDDEE AABBCCDDEE -> (world_size = 3, num_groups = 3, chunk_size = 2) -> [AABB AABB AABB, CCDD CCDD CCDD, EE EE EE]"""

    def __init__(self, world_size: int, dim: int, num_groups: int, chunk_size: int) -> None:
        super().__init__(world_size, dim)
        self.num_groups = num_groups
        self.chunk_size = chunk_size

    def __call__(self, tensor: Tensor, rank: int) -> Tensor:
        shape = list(tensor.shape)  # ... x hidden_size x ...
        shape[self.dim] //= self.num_groups
        shape.insert(self.dim, self.num_groups)  # ... group x group_size x ...
        shape[self.dim + 1] //= self.chunk_size
        shape.insert(self.dim + 2, self.chunk_size)  # ... group x chunk x chunk_size ...
        return (
            tensor.reshape(shape).tensor_split(self.world_size, dim=self.dim + 1)[rank].flatten(self.dim, self.dim + 2)
        )

    def undo(self, tensors: Sequence[Tensor]) -> Tensor:
        grouped_tensor = []
        for tensor in tensors:
            shape = list(tensor.shape)  # ... x hidden_size x ...
            shape[self.dim] = shape[self.dim] // self.num_groups
            shape.insert(self.dim, self.num_groups)  # ... group x group_size x ...
            grouped_tensor.append(tensor.reshape(*shape).cpu())

        return torch.cat(grouped_tensor, dim=self.dim + 1).flatten(self.dim, self.dim + 1)
