"""
The main TensorParallel module wrapper
"""
import logging
import os
import threading
from contextlib import nullcontext
from typing import Any, Optional, Sequence, Union

import torch
from torch import nn
from torch._utils import ExceptionWrapper, _get_all_device_indices, _get_device_index
from torch.cuda.amp import autocast
from torch.nn.parallel import parallel_apply

from tensor_parallel.cross_device_ops import broadcast_coalesced
from tensor_parallel.slicer_wrapper import TENSOR_PARALLEL_USE_NATIVE, Config
from tensor_parallel.utils import nested_flatten, nested_pack

logger = logging.getLogger(__file__)


class TensorParallel(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        device_ids: Optional[Sequence[torch.device]] = None,
        output_device: Optional[torch.device] = None,
        output_device_index: Optional[int] = None,
        config: Optional[Config] = None,
    ):
        super().__init__()
        original_params = sum(p.numel() for p in module.parameters())
        assert output_device is None or output_device_index is None, "please specify either device or index, not both"
        device_ids = check_device_ids(device_ids)

        if output_device is not None:
            output_device = canonicalize_device(output_device)
            assert output_device in device_ids, f"Output device {output_device} not in {device_ids}"
            output_device_index = device_ids.index(output_device)
            del output_device
        elif output_device_index is None:
            output_device_index = 0

        self.module_shards = nn.ModuleList()

        self.devices = device_ids
        self.output_device_index = output_device_index
        self.all_cuda = all(device.type == "cuda" for device in self.devices)
        self.device_ids = [_get_device_index(x, optional=True, allow_cpu=True) for x in device_ids]
        world_size = len(self.devices)

        if len(device_ids) <= 1:
            self.module_shards.append(module)
            return

        if config is None:
            config = Config.get_default_config(module)
            logger.info("Using automatic config: sharding individual linear/conv/emb layers")

        config_with_ops = config.create_collective_ops(self.devices)
        # ^-- creates a copy of comfig with collective op instances, such as AllReduce and AllGather

        for rank, device in enumerate(self.devices):
            self.module_shards.append(
                config.make_shard(module, device, config_with_ops, rank=rank, world_size=world_size)
            )

        # self-diagnostics: check if the model was sharded properly

        params_per_shard = [sum(p.numel() for p in shard.parameters()) for shard in self.module_shards]
        assert sum(params_per_shard) >= original_params, "Internal assert failed: lost some parameters during sharding"
        self.param_fractions = tuple(params_i / original_params for params_i in params_per_shard)
        inefficiency_rate = (sum(self.param_fractions) - 1) / len(device_ids)  # extra params rate per GPU
        log_level = logging.DEBUG if inefficiency_rate < 0.1 else logging.WARNING
        logger.log(
            log_level,
            f"Inefficiency warning: model has {original_params} params but shards have {params_per_shard} params. "
            f"This means that each device uses {inefficiency_rate * 100:.3f}% extra memory for parameters",
        )

    def forward(self, *args, **kwargs):
        if len(self.module_shards) <= 1:
            return [self.module_shards[0](*args, **kwargs)][self.output_device_index]
        args_and_kwargs = (args, kwargs)
        flat_tensors = [obj for obj in nested_flatten(args_and_kwargs) if isinstance(obj, torch.Tensor)]
        flat_tensors_replicated = broadcast_coalesced(flat_tensors, self.devices, all_cuda=self.all_cuda)
        next_tensor_index = 0
        args_and_kwargs_replicated = [list() for _ in self.device_ids]
        for obj in nested_flatten(args_and_kwargs):
            if isinstance(obj, torch.Tensor):
                for idx in range(len(self.module_shards)):
                    args_and_kwargs_replicated[idx].append(flat_tensors_replicated[idx][next_tensor_index])
                next_tensor_index += 1
            else:
                for idx in range(len(self.module_shards)):
                    args_and_kwargs_replicated[idx].append(obj)
        for idx in range(len(self.module_shards)):
            args_and_kwargs_replicated[idx] = nested_pack(args_and_kwargs_replicated[idx], args_and_kwargs)
        inputs, kwargs_tup = zip(*args_and_kwargs_replicated)
        if self.all_cuda and not TENSOR_PARALLEL_USE_NATIVE:
            return parallel_apply(self.module_shards, inputs, kwargs_tup, self.devices)[self.output_device_index]
        else:
            return parallel_apply_simple(self.module_shards, inputs, kwargs_tup, self.devices)[self.output_device_index]


def parallel_apply_simple(
    modules: Sequence[nn.Module],
    inputs: Sequence[Sequence[torch.Tensor]],
    kwargs_tup: Optional[Any],
    devices: Sequence[torch.device],
) -> Sequence[Sequence[torch.Tensor]]:
    r"""a version of parallel_apply that does not use cuda streams; somewhat slower"""
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    lock = threading.Lock()
    results = {}
    grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()

    def _worker(i, module, input, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            device_ctx = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
            with device_ctx, autocast(enabled=autocast_enabled):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        threads = [
            threading.Thread(target=_worker, args=(i, module, input, kwargs, device))
            for i, (module, input, kwargs, device) in enumerate(zip(modules, inputs, kwargs_tup, devices))
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs


def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None


def canonicalize_device(device: Union[torch.device, str]) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device(device, index=0)
    return device


def check_device_ids(device_ids: Optional[Sequence[torch.device]]) -> Sequence[torch.device]:
    if device_ids is None:
        device_ids = _get_all_device_indices() if torch.cuda.is_available() else []
    return tuple(map(canonicalize_device, device_ids))


class PerDeviceTensors:
    """tensors located on different deviecs that will *not* be broadcasted when passed to TensorParallel.forward"""

    def __init__(self, *tensors: torch.Tensor):
        # note: this will not be broadcasted because broadcast_coalesced does not broadcast class properties
        self.tensors = tuple(tensors)

    def __getitem__(self, i: int):
        return self.tensors[i]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.tensors})"
