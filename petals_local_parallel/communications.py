from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.distributed as dist

import threading

class Communicator(ABC):
    @abstractmethod
    def all_reduce(self, x: Tensor) -> Tensor:
        pass


class TorchrunCommunicator(Communicator):
    def __init__(self) -> None:
        super().__init__()

    def all_reduce(self, x: Tensor, rank: int) -> Tensor:
        dist.all_reduce(x)
        return x


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


TENSOR_PARALLEL_COMMUNICATOR: Communicator = None # is set during model wrapping
