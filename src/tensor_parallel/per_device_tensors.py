import torch


class PerDeviceTensors:
    """tensors located on different deviecs that will *not* be broadcasted when passed to TensorParallel.forward"""

    def __init__(self, *tensors: torch.Tensor):
        # note: this will not be broadcasted because broadcast_coalesced does not broadcast class properties
        self.tensors = tuple(tensors)

    def __getitem__(self, i: int):
        return self.tensors[i]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.tensors})"

    @property
    def shape(self):
        return self.tensors[0].shape

    def size(self, dim: int):
        return self.tensors[0].size(dim)
