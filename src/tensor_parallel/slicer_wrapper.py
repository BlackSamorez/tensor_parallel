"""
Tensor parallelism config and functions for splitting model into shards
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
from copy import deepcopy
from functools import partial
from itertools import chain
from typing import Callable, Dict, Sequence, Union

import torch
from torch import nn
from torch.nn.modules import conv

import tensor_parallel.cross_device_ops as cross_device_ops
from tensor_parallel.communications import (
    AllGather,
    AllReduce,
    DistributedAllGather,
    DistributedAllReduce,
    NCCLAllGather,
    NCCLAllReduce,
)

Arg = Union[int, str]
Pattern = Union[str, re.Pattern]
Action = Union[str, Callable]  # actions describe what to do with tensors
StateRules = Dict[Pattern, Action]  # state rules are pattern-matched actions on module state dict
ModuleRules = Dict[Pattern, Dict[Arg, Action]]  # module rules are pattern-matched actions on inputs/outputs/attrs

logger = logging.getLogger(__file__)


TENSOR_PARALLEL_USE_NATIVE = bool(os.environ.get("TENSOR_PARALLEL_USE_NATIVE"))


@dataclasses.dataclass
class Config:
    state_rules: StateRules
    input_rules: ModuleRules
    output_rules: ModuleRules
    attr_rules: ModuleRules

    def __init__(
        self, state_rules: StateRules, input_rules: ModuleRules, output_rules: ModuleRules, attr_rules: ModuleRules
    ):
        all_rules = [state_rules, input_rules, output_rules, attr_rules]
        for i, rule_set in enumerate(all_rules):
            all_rules[i] = dict((re.compile(pattern), actions) for pattern, actions in rule_set.items())
        self.state_rules, self.input_rules, self.output_rules, self.attr_rules = all_rules

    @classmethod
    def get_default_config(cls, module: nn.Module) -> Config:
        """Make a generic config that wraps individual linear, embedding and convolutional layers"""
        config = cls({}, {}, {}, {})
        emb_weights = {m.weight for m in module.modules() if isinstance(m, (nn.Embedding, nn.EmbeddingBag))}

        for name, module in module.named_modules():
            if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
                assert module.max_norm is None or module.norm_type < 2
                assert getattr(module, "bias", None) is None or module.bias.shape == module.embedding_dim
                config.state_rules[f"^{name}.weight$"] = "split 1"
                if hasattr(module, "bias"):
                    config.state_rules[f"^{name}.bias$"] = "split 0"
                config.output_rules[f"^{name}$"] = {0: "gather -1"}
            elif isinstance(module, nn.Linear):
                assert module.weight.shape == (module.out_features, module.in_features)
                assert module.bias is None or module.bias.shape == (module.out_features,)
                if module.weight not in emb_weights:  # regular linear layer
                    config.state_rules[f"^{name}.(weight|bias)$"] = "split 0"
                    config.output_rules[f"^{name}$"] = {0: "gather -1"}
                else:
                    # linear weight tied with embeddings; this is a popular special case for language models;
                    # since embedding weight will be sliced over dim 1, we should adapt to the input-sliced weight
                    config.input_rules[f"^{name}$"] = {0: "split -1"}
                    config.output_rules[f"^{name}$"] = {0: "sum"}
                    if module.bias is not None:
                        config.state_rules[f"^{name}.bias$"] = "scale"
            elif isinstance(module, conv._ConvNd) and module.groups == 1:
                shape = [module.out_channels, module.in_channels] + list(module.kernel_size)
                shape[:2] = shape[:2][::-1] if module.transposed else shape[:2]
                shape = tuple(shape)
                assert module.weight.shape == shape, f"{module.weight.shape} != {shape}"
                assert module.bias is None or module.bias.shape == (module.out_channels,), module.bias.shape
                config.state_rules[f"^{name}.weight$"] = "split 1" if module.transposed else "split 0"
                config.state_rules[f"^{name}.bias$"] = "split 0"
                config.output_rules[f"^{name}$"] = {0: "gather 1"}
            elif isinstance(module, conv._ConvNd) and module.groups != 1:
                logger.warning(
                    f"AutoConfig does not support sharding convolution layers {name} with {module.groups} groups yet; "
                    "If you're sure that you need it, implement sharding in place of this message and open a PR"
                )
        return config

    def create_collective_ops(self, devices):
        """
        Return a copy of config with thread-parallel collective operations, such as AllGather and AllReduce

        :note: this function should be called during TensorParallel init, before making shards
        """
        return dataclasses.replace(
            self,
            input_rules=create_collective_ops(self.input_rules, devices),
            output_rules=create_collective_ops(self.output_rules, devices),
        )

    def make_shard(
        self, module: nn.Module, device: torch.device, config: Config, *, rank: int, world_size: int
    ) -> nn.Module:
        """Apply a tensor-parallel config to a high-level module, return module shard for a given rank and device"""
        assert (
            len(list(module.children())) != 0
        ), "Please ensure module is a container (e.g. Sequential), not a single layer"
        shard = deepcopy(module).to(device)
        del module

        # convert parameters and buffers
        process_state_(shard, config, rank=rank, world_size=world_size)

        # convert or wrap intermediate modules
        unique_wrappers = {}
        with torch.no_grad():
            for name, submodule in shard.named_modules():
                if submodule in unique_wrappers:
                    continue  # wrap a module at most once
                maybe_wrapped = config._maybe_wrap_submodule(name, submodule, rank=rank, world_size=world_size)
                if maybe_wrapped:  # apply the same wrapper to all occurrences of the given module
                    unique_wrappers[submodule] = maybe_wrapped

        for parent in list(shard.modules()):
            for child_name, child in list(parent.named_children()):
                if child in unique_wrappers:
                    setattr(parent, child_name, unique_wrappers[child])

        return unique_wrappers.get(shard, shard)  # wrap the root module if needed

    def make_distributed_shard(self, module: nn.Module, device: torch.device):
        config_with_ops = self.create_collective_ops([torch.device("cpu")] * torch.distributed.get_world_size())
        return self.make_shard(
            module,
            device,
            config_with_ops,
            rank=torch.distributed.get_rank(),
            world_size=torch.distributed.get_world_size(),
        )

    def _maybe_wrap_submodule(self, name: str, module: nn.Module, *, rank: int, world_size: int) -> nn.Module:
        """
        Apply the tensor parallelism config to the specified module, return wrapped module

        :note: this function does NOT apply state dict changes (state_rules), these should be applied separately!
        :note: this function will modify the original module in-place!
        """
        attr_actions = find_matching_actions("attr", name, self.attr_rules)
        input_actions = find_matching_actions("input", name, self.input_rules)
        output_actions = find_matching_actions("output", name, self.output_rules)
        if attr_actions:
            process_attrs_(module, attr_actions, rank=rank, world_size=world_size)
        if input_actions or output_actions:
            module = _TensorParallelWrapper(module, input_actions, output_actions, rank=rank, world_size=world_size)
        return module


def find_matching_actions(action_type: str, name: str, rules: ModuleRules) -> Dict[Arg, Action]:
    found_actions = {}
    for pattern, actions_for_pattern in rules.items():
        if pattern.search(name) is not None:
            for key, action in actions_for_pattern.items():
                found_actions[key] = action
    return found_actions


def apply_action(input: torch.Tensor, action: Action, *, rank: int, world_size: int):
    if callable(action):
        return action(input, rank=rank)  # allreduce/allgather or a custom user-defined callable
    action_type, *opts = action.split()
    if action_type == "split":
        dim = int(opts[0])
        return torch.tensor_split(input, world_size, dim=dim)[rank]
    if action_type == "split_with_sizes":
        assert len(opts) == 3 and all(map(str.isdigit, opts))
        dim = int(opts[0])
        return torch.tensor_split(input, world_size, dim=dim)[rank]
    if action_type == "scale":
        return input / world_size
    if action_type == "scale_int":
        assert input % world_size == 0
        return input // world_size
    raise Exception(f"unexpected action {action_type}; supported actions: split, scale, or custom user-defined")


def create_collective_ops(rules: dict, devices: Sequence[torch.device]):
    """Initialize collective thread-parallel operations from config rules"""
    world_size = len(devices)
    all_cuda = all(device.type == "cuda" for device in devices)
    unique_output_transforms = {op for output_actions in rules.values() for op in output_actions.values()}
    transform_map = {}
    if torch.distributed.is_initialized():
        make_allreduce, make_allgather = DistributedAllReduce, DistributedAllGather
    elif all_cuda and not TENSOR_PARALLEL_USE_NATIVE:
        make_allreduce, make_allgather = NCCLAllReduce, NCCLAllGather
    else:
        make_allreduce = partial(
            AllReduce,
            reduce_op=lambda xs, destination: cross_device_ops.reduce_add(xs, destination, all_cuda=all_cuda),
            gather_op=lambda xs, destination: cross_device_ops.gather(
                xs, dim=0, destination=destination, all_cuda=all_cuda
            ),
        )
        make_allgather = lambda world_size, dim: AllGather(
            world_size, gather_op=lambda xs, destination: cross_device_ops.gather(xs, dim=dim, all_cuda=all_cuda)
        )

    for transform in unique_output_transforms:
        if callable(transform):
            continue  # user-defined transform, no action needed
        transform_type, *opts = transform.split()
        if transform_type in ("split", "scale", "scale_int"):
            continue  # not a collective op, no action needed

        if transform_type == "sum":
            transform_map[transform] = make_allreduce(world_size)
        elif transform_type == "gather":
            dim = int(opts[0]) if opts else -1
            transform_map[transform] = make_allgather(world_size, dim)

    initialized_output_rules = {}
    for pattern, output_actions in rules.items():
        output_actions = {key: transform_map.get(rule, rule) for key, rule in output_actions.items()}
        initialized_output_rules[pattern] = output_actions
    return initialized_output_rules


@torch.no_grad()
def process_state_(module: nn.Module, config: Config, *, rank: int, world_size: int):
    """Modify module parameters and/or buffers in-place"""
    unused_patterns = set(config.state_rules.keys())
    for name, param in chain(module.named_parameters(), module.named_buffers()):
        for pattern, action in config.state_rules.items():
            if pattern.search(name) is not None:
                param.data = apply_action(param.data, action, rank=rank, world_size=world_size).clone()
                # note: .clone is required so that the resulting parameter owns its storage
                unused_patterns.discard(pattern)

    if unused_patterns:
        logger.warning(f"The following patterns in state_rules were unused: {[str(p) for p in unused_patterns]}")


def process_attrs_(module: nn.Module, actions: Dict[Arg, str], rank: int, world_size: int):
    """Modify module properties in-place"""
    assert not getattr(module, "__tensor_parallel_process_attrs", False), "process_attrs was called more than once"
    for attr, action in actions.items():
        setattr(module, attr, apply_action(getattr(module, attr), action, rank=rank, world_size=world_size))
    module.__tensor_parallel_process_attrs = True


def process_input(input_actions: Dict[Arg, str], rank: int, world_size: int, *args, **kwargs):
    extended_kwargs = dict(kwargs)
    extended_kwargs.update(enumerate(args))
    for target, action in input_actions.items():
        extended_kwargs[target] = apply_action(extended_kwargs.get(target), action, rank=rank, world_size=world_size)
    args = [extended_kwargs.pop(i) for i in range(len(args))]
    return args, extended_kwargs


def process_output(
    output, output_actions: Dict[Arg, Callable[[torch.Tensor, int], torch.Tensor]], *, rank: int, world_size: int
):
    if isinstance(output, torch.Tensor):
        return process_output({0: output}, output_actions, rank=rank, world_size=world_size)[0]
    if isinstance(output, Sequence):
        output_dict = process_output(dict(enumerate(output)), output_actions, rank=rank, world_size=world_size)
        return type(output)((output_dict[i] for i in range(len(output))))
    for target, action in output_actions.items():
        output[target] = apply_action(output.get(target), action, rank=rank, world_size=world_size)
    return output


class _TensorParallelWrapper(nn.Module):
    """Wraps a single module, applies tensor parallelism actions to module inputs and outputs"""

    def __init__(
        self,
        module: nn.Module,
        input_actions: Dict[Arg, Action],
        output_actions: Dict[Arg, Action],
        *,
        rank: int,
        world_size: int,
    ):
        super().__init__()
        self.module = module
        self.input_actions, self.output_actions = input_actions, output_actions
        self.rank, self.world_size = rank, world_size

    def forward(self, *args, **kwargs):
        args, kwargs = process_input(self.input_actions, self.rank, self.world_size, *args, **kwargs)
        output = self.module(*args, **kwargs)
        return process_output(output, self.output_actions, rank=self.rank, world_size=self.world_size)
