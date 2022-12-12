from abc import ABC, abstractmethod
import threading

import torch
from torch import Tensor
import torch.distributed as dist

from .all_to_all_communication_primitives import AllReduce, AllGather

class Communicator(ABC):
    @abstractmethod
    def all_reduce(self, x: Tensor, rank: int) -> Tensor:
        pass

    @abstractmethod
    def all_gather(self, x: Tensor, rank: int) -> Tensor:
        pass


class TorchrunCommunicator(Communicator):
    def __init__(self) -> None:
        super().__init__()

    def all_reduce(self, x: Tensor, rank: int) -> Tensor:
        dist.all_reduce(x)
        return x

    def all_gather(self, x: Tensor, rank: int) -> Tensor:
        targets = [torch.zeros_like(x) for _ in dist.get_world_size()]
        dist.all_gather(targets, x)
        return torch.cat(targets)


class CentralizedCommunicator(Communicator):
    def __init__(self, devices) -> None:
        super().__init__()

        self.main_device = devices[0]
        self.num_workers = len(devices)
        self.num_contributions = 0
        self.cv = threading.Condition()
        self.done = False

        self.buffer: torch.Tensor = None

    def all_reduce(self, x: Tensor, rank: int) -> Tensor:
        with self.cv:
            if self.done:
                while self.done:
                    self.cv.wait()
            
            if self.num_contributions == 0:
                self.buffer = torch.zeros_like(x, device=self.main_device)
            self.buffer += x.to(self.main_device)
            self.num_contributions += 1

            if self.num_contributions == self.num_workers:
                self.done = True
                self.num_contributions -= 1
                self.cv.notify_all()
                return self.buffer.clone().to(x.device)

            while not self.done:
                self.cv.wait()

            self.num_contributions -= 1
            if self.num_contributions == 0:
                self.done = False
            return self.buffer.clone().to(x.device)

    def all_gather(self, x: Tensor, rank: int) -> Tensor:
        raise NotImplementedError("CentralizedCommunicator does not support all gather (yet?)")


class AllToAllCommunicator(Communicator):
    def __init__(self, world_size: int):
        self.all_reduce = AllReduce(world_size)
        self.all_gather = AllGather(world_size)

    def all_reduce(self, x: Tensor, rank: int) -> Tensor:
        return self.all_reduce(x, rank)

    def all_gather(self, x: Tensor, rank: int) -> Tensor:
        return self.all_gather(x, rank)


TENSOR_PARALLEL_COMMUNICATOR: Communicator = None # is set during model wrapping
