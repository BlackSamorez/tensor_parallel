from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.distributed as dist

class Communicator(ABC):
    @abstractmethod
    def all_reduce(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def all_gather(self, x: Tensor) -> Tensor:
        pass


class TorchrunCommunicator(Communicator):
    def __init__(self) -> None:
        super().__init__()

    def all_reduce(self, x: Tensor) -> Tensor:
        dist.all_reduce(x)
        return x

    def all_gather(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            gathering = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(gathering, x)
        gathering[dist.get_rank()] = x
        return torch.cat(gathering, dim=1)


def get_optimal_communicator() -> Communicator:
    if dist.is_initialized():
        return TorchrunCommunicator()

    raise NotImplementedError("No communicators for you")


TENSOR_PARALLEL_COMMUNICATOR: Communicator = None # is set during model wrapping
