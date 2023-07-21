"""
The main TensorParallel module wrapper
"""
import logging
import threading
from contextlib import nullcontext
from typing import Any, Optional, Sequence, Union

import torch
from torch import nn
from torch._utils import ExceptionWrapper, _get_all_device_indices, _get_device_index
from torch.cuda.amp import autocast
from torch.nn.parallel import parallel_apply

from tensor_parallel.autoconfig import get_default_config
from tensor_parallel.config import TENSOR_PARALLEL_USE_NATIVE, Config, add_lora_rules
from tensor_parallel.cross_device_ops import broadcast_coalesced
from tensor_parallel.shard import make_shard
from tensor_parallel.utils import nested_flatten, nested_pack

logger = logging.getLogger(__file__)


class TensorParallel(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        device_ids: Optional[Sequence[torch.device]] = None,
        output_device: Optional[torch.device] = None,
        output_device_index: Optional[int] = None,
        tensor_parallel_config: Optional[Config] = None,
        delay_init: bool = False,
        distributed: bool = True,
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
        self.need_delayed_init = delay_init
        world_size = len(self.devices)

        if len(device_ids) <= 1:
            self.module_shards.append(module)
            if len(device_ids) == 1 and not delay_init:
                self.module_shards[0].to(device_ids[0])
            return

        if tensor_parallel_config is None:
            tensor_parallel_config = get_default_config(module, self.devices)
            logger.info("Using automatic config: sharding individual linear/conv/emb layers")

        tensor_parallel_config = add_lora_rules(module, tensor_parallel_config)
        self.tensor_parallel_config = tensor_parallel_config

        config_with_ops = tensor_parallel_config.create_collective_ops(self.devices, distributed)
        # ^-- creates a copy of comfig with collective op instances, such as AllReduce and AllGather

        for rank, device in enumerate(self.devices):
            if delay_init:
                device = torch.device("cpu")
            self.module_shards.append(make_shard(module, device, config_with_ops, rank=rank, world_size=world_size))

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

        # more self-diagnostics: make sure that the model was not cast .to one device
        self._sanity_check_params = nn.ParameterList(
            [nn.Parameter(torch.empty(0, device=device), requires_grad=False) for device in self.devices]
        )
        self.preserve_shards_when_saving: bool = True

    def prepare_args_kwargs_for_forward(self, *args, **kwargs):
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
        return zip(*args_and_kwargs_replicated)

    def forward(self, *args, **kwargs):
        if self.need_delayed_init:
            for shard, device in zip(self.module_shards, self.devices):
                shard.to(device)
            self.need_delayed_init = False

        if len(self.module_shards) <= 1:
            return [self.module_shards[0](*args, **kwargs)][self.output_device_index]

        if not all(p.device == d for p, d in zip(self._sanity_check_params, self.devices)):
            raise ValueError(
                "Model parameters were moved to incorrect devices, did call on model.cuda() or "
                "model.to(device)? If so, please avoid doing that"
            )
        inputs, kwargs_tup = self.prepare_args_kwargs_for_forward(*args, **kwargs)
        if self.all_cuda and not TENSOR_PARALLEL_USE_NATIVE:
            return parallel_apply(self.module_shards, inputs, kwargs_tup, self.devices)[self.output_device_index]
        else:
            return parallel_apply_simple(self.module_shards, inputs, kwargs_tup, self.devices)[self.output_device_index]

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        if self.preserve_shards_when_saving:
            return state_dict

        for i in range(len(self.module_shards)):
            sanity_check_param_name = next(
                name for name, _ in state_dict.items() if name.endswith(f"_sanity_check_params.{i}")
            )
            del state_dict[sanity_check_param_name]

        # fix names for zero-3'ed params that were inside _TensorParallelWrapper
        names_inside_tp_wrapper = [name for name in state_dict.keys() if "tp_wrapped_module." in name]
        for name in names_inside_tp_wrapper:
            state_dict[name.replace("tp_wrapped_module.", "")] = state_dict.pop(name)

        try:
            shards_prefix = next(name for name, _ in state_dict.items() if "module_shards." in name)
        except StopIteration:
            return state_dict  # no parameters are actually tensor parallel
        shards_prefix = shards_prefix[: shards_prefix.find("module_shards.") + len("module_shards.")]
        module_prefix = shards_prefix[: -len("module_shards.")]

        # get names for desired tensors and where to find them (shards of zero-3)
        is_name_prefixed = {}
        for name, tensor in state_dict.items():
            if name.startswith(shards_prefix + "0."):  # dict entry is from shards
                is_name_prefixed[name[len(shards_prefix) + 2 :]] = True
            if not name.startswith(shards_prefix):  # dict entry is from zero-3
                is_name_prefixed[name[len(module_prefix) :]] = False

        for unsharded_name, is_prefixed in is_name_prefixed.items():
            for pattern, action in self.tensor_parallel_config.state_rules.items():
                if pattern.search(unsharded_name) is not None:
                    tensor_shards = {
                        name: tensor for name, tensor in state_dict.items() if name.endswith(unsharded_name)
                    }
                    tensor_shards = dict(sorted(tensor_shards.items()))  # basically sort by shard number
                    state_dict[module_prefix + unsharded_name] = action.undo(list(tensor_shards.values()))
                    break
            else:
                state_dict[module_prefix + unsharded_name] = next(
                    tensor for name, tensor in state_dict.items() if name.endswith(unsharded_name)
                )
            if is_prefixed:
                # delete sharded tensor entries
                for i in range(len(self.module_shards)):
                    del state_dict[f"{shards_prefix}{i}.{unsharded_name}"]

        return state_dict


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
        device = torch.device(device.type, index=0)
    return device


def check_device_ids(device_ids: Optional[Sequence[torch.device]]) -> Sequence[torch.device]:
    if device_ids is None:
        device_ids = _get_all_device_indices() if torch.cuda.is_available() else []
    return tuple(map(canonicalize_device, device_ids))
