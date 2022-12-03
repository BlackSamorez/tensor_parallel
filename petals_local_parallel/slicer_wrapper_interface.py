from slicer_wrapper import SlicingConfig, get_tensor_parallel_model_slice, MultithreadedModule
from slicing_configs import SLICING_CONFIGS
import communications

import torch
import torch.distributed as dist


def tensor_parallel(model_cls, slicing_config: SlicingConfig = None, devices=None):
    if slicing_config is None:
        try:
            slicing_config = SLICING_CONFIGS[model_cls.__name__]
        except KeyError:
            raise NotImplemented(f"Unknown model type {model_cls.__name__} and lazy mode not implemented yet. Must specify config")

    if dist.is_initialized():
        communications.TENSOR_PARALLEL_COMMUNICATOR = communications.TorchrunCommunicator()
        return get_tensor_parallel_model_slice(model_cls, slicing_config, dist.get_rank(), dist.get_world_size()) # each torchrun process only need one slice
    else:
        assert(devices is not None), "devices must be provided when using tensor_parallel without torchrun"
        communications.TENSOR_PARALLEL_COMMUNICATOR = communications.CentralizedCommunicator(devices) # TODO: this is for tests so make it more obscure
        slices = []
        for i in range(len(devices)):
            slices.append(get_tensor_parallel_model_slice(model_cls, slicing_config, i, len(devices)))
        return MultithreadedModule(slices, devices)
