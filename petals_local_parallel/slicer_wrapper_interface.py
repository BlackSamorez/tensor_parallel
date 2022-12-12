from .slicing_config import SlicingConfig
from .slicer_wrapper import get_tensor_parallel_model_slice, MultithreadedModule
from .slicing_configs import SLICING_CONFIGS
from . import communications

import torch
import torch.distributed as dist


def tensor_parallel(model_cls, devices, slicing_config: SlicingConfig = None):
    if slicing_config is None:
        try:
            slicing_config = SLICING_CONFIGS[model_cls.__name__]
        except KeyError:
            print("No slicing_config provided. Using nn.Linear slicing fallback")

    if dist.is_initialized():
        communications.TENSOR_PARALLEL_COMMUNICATOR = communications.TorchrunCommunicator()
        rank = dist.get_rank()
        return get_tensor_parallel_model_slice(model_cls, slicing_config, rank, dist.get_world_size()) # each torchrun process only need one slice
    else:
        assert(devices is not None), "devices must be provided when using tensor_parallel without torchrun"
        communications.TENSOR_PARALLEL_COMMUNICATOR = communications.AllToAllCommunicator(len(devices))
        slices = []
        for i in range(len(devices)):
            slices.append(get_tensor_parallel_model_slice(model_cls, slicing_config, i, len(devices)))
        return MultithreadedModule(slices, devices)
