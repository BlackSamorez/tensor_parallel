import re
from typing import Callable, Dict, Iterator, Tuple, Union

import torch
import torch.nn as nn
from .autoconfig import build_default_slicing_config
from .all_to_all_communication_primitives import AllReduce, AllGather
from .slicing_config import SlicingConfig
from torch.nn.parallel import parallel_apply
from transformers import PreTrainedModel, PretrainedConfig


Pattern, Arg = str, Union[int, str]


def slice_weight_vertical(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    return tensor.tensor_split(world_size, dim=-2)[rank]


def slice_bias_vertical(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    return tensor.tensor_split(world_size)[rank]


def slice_weight_horizontal(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    return tensor.tensor_split(world_size, dim=-1)[rank]


def slice_bias_horizontal(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    return tensor / world_size


SLICING_RULES = {
    ("vertical", "weight"): slice_weight_vertical,
    ("vertical", "bias"): slice_bias_vertical,
    ("horizontal", "weight"): slice_weight_horizontal,
    ("horizontal", "bias"): slice_bias_horizontal,
}


def slice_tensors(
    key_parameter_iterator: Iterator[Tuple[str, nn.Parameter]], tensor_rules: Dict[Arg, str], rank: int, world_size: int
):
    regular_rules = [(re.compile(key), value) for key, value in tensor_rules.items()]

    with torch.no_grad():
        for name, param in key_parameter_iterator:
            for pattern, rule in regular_rules:
                if pattern.search(name) is not None:
                    name_ending = name.split(".")[-1]
                    param.data = SLICING_RULES[rule, name_ending](param.data, rank=rank, world_size=world_size)


def process_input(rules: Dict[Arg, str], rank: int, world_size: int, *args, **kwargs):
    extended_kwargs = dict(kwargs)
    extended_kwargs.update(enumerate(args))
    for target, action in rules.items():
        if not isinstance(extended_kwargs.get(target), torch.Tensor):
            continue  # optional parameter is None or False
        action_type, *opts = action.split()
        if action_type == "cut":
            dim = int(opts[0])
            extended_kwargs[target] = extended_kwargs[target].tensor_split(world_size, dim=dim)[rank]
        elif action_type == "scale":
            extended_kwargs[target] = extended_kwargs[target] / world_size
        else:
            raise Exception(f"unexpected action {action_type}")

    args = [extended_kwargs.pop(i) for i in range(len(args))]
    return args, extended_kwargs


def process_output(output, rules: Dict[Arg, Callable[[torch.Tensor, int], torch.Tensor]], rank: int):
    if isinstance(output, torch.Tensor):
        return process_output([output], rules, rank)[0]
    for target, action in rules.items():
        output[target] = action(output[target], rank)
    return output


def process_attr(module: nn.Module, rules: Dict[Arg, str], rank: int, world_size: int):
    for attr, action in rules.items():
        if action == "scale_int":
            setattr(module, attr, getattr(module, attr) // world_size)
        else:
            raise NotImplementedError(action)


class ParallelLayerWrapper(nn.Module):
    def __init__(self, module: nn.Module, module_rules: dict, rank: int, world_size: int):
        super().__init__()
        self.module = module
        process_attr(self.module, module_rules["attributes"], rank=rank, world_size=world_size)

        self.input_rules = module_rules["input"]
        self.output_rules = module_rules["output"]

        self.rank = rank
        self.world_size = world_size

    def forward(self, *args, **kwargs):
        args, kwargs = process_input(self.input_rules, self.rank, self.world_size, *args, **kwargs)
        output = self.module(*args, **kwargs)
        return process_output(output, self.output_rules, self.rank)


def wrap_submodules(model: nn.Module, module_rules: dict, rank: int, world_size: int):
    unique_output_transforms = {op for rules in module_rules.values() for op in rules['output'].values()}
    transform_map = {}
    for transform in unique_output_transforms:
        if transform == 'sum':
            transform_map[transform] = AllReduce(world_size, reduce_op=sum)
        elif transform == 'gather':
            transform_map[transform] = AllGather(world_size, gather_op=lambda xs: torch.cat(xs, dim=-1))
        elif callable(transform):
            transform_map[transform] = transform  # user-defined transform, no action needed
        else:
            raise NotImplementedError(f"Unknown output transform {transform}")

    for rules in module_rules.values():
        for key, rule in rules['output'].items():
            rules['output'][key] = transform_map[rule]

    unique_wrappers = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            for pattern, rule in module_rules.items():
                if re.search(pattern, name) is not None:
                    unique_wrappers[module] = ParallelLayerWrapper(module, rule, rank=rank, world_size=world_size)

    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if child in unique_wrappers:
                setattr(parent, child_name, unique_wrappers[child])


def get_tensor_parallel_model_slice(model_cls, slicing_config: SlicingConfig, rank: int, world_size: int):
    class _TensorParallelSlice(model_cls):
        def __new__(cls, *args, __slicing_config=slicing_config, __rank=rank, __world_size=world_size, **kwargs):
            _TensorParallelSlice.slicing_config = __slicing_config
            _TensorParallelSlice.rank = __rank
            _TensorParallelSlice.world_size = __world_size

            model = model_cls(*args, **kwargs)  # Create an instance of vanilla model

            if _TensorParallelSlice.slicing_config is None:
                _TensorParallelSlice.slicing_config = build_default_slicing_config(model)

            # modify untrained parameters/buffers
            slice_tensors(model.named_parameters(), _TensorParallelSlice.slicing_config.tensor_rules, rank, world_size)
            return model

        @classmethod
        def _load_pretrained_model(cls, model: model_cls, state_dict, loaded_keys, *args, **kwargs):
            slice_tensors(
                state_dict.items(),
                _TensorParallelSlice.slicing_config.tensor_rules,
                _TensorParallelSlice.rank,
                _TensorParallelSlice.world_size,
            )
            result = super()._load_pretrained_model(model, state_dict, loaded_keys, *args, **kwargs)

            wrap_submodules(
                model,
                _TensorParallelSlice.slicing_config.module_rules,
                _TensorParallelSlice.rank,
                _TensorParallelSlice.world_size,
            )

            return result

    return _TensorParallelSlice


class MultithreadedModule(PreTrainedModel):
    def __init__(self, slice_types, devices) -> None:
        super().__init__(PretrainedConfig())  # Temporary empty config. Gets replaced in from_pretrained
        assert len(slice_types) == len(devices)
        self.slice_types = slice_types
        self.slices = torch.nn.ModuleList()
        self.devices = devices

    def forward(self, *args, **kwargs):
        def scatter_map(obj):
            if isinstance(obj, torch.Tensor):
                return [obj.clone().to(targets) for targets in self.devices]
            if isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields"):
                return [type(obj)(*args) for args in zip(*map(scatter_map, obj))]
            if isinstance(obj, tuple) and len(obj) > 0:
                return list(zip(*map(scatter_map, obj)))
            if isinstance(obj, list) and len(obj) > 0:
                return [list(i) for i in zip(*map(scatter_map, obj))]
            if isinstance(obj, dict) and len(obj) > 0:
                return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]
            return [obj for targets in self.devices]

        inputs = scatter_map(args)
        kwargs_tup = scatter_map(kwargs)

        return parallel_apply(self.slices, inputs, kwargs_tup=kwargs_tup)[0]

    def from_pretrained(self, *args, **kwargs):
        self.slices = torch.nn.ModuleList(
            [slice_type.from_pretrained(*args, **kwargs) for slice_type in self.slice_types]
        )
        self.config = self.slices[0].config
        return self.scatter()

    def scatter(self):
        for slice, device in zip(self.slices, self.devices):
            slice.to(device, non_blocking=True)
        return self
