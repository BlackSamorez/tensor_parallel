import dataclasses
import logging
import os
import re
from functools import partial
from typing import Any, Callable, Dict, Sequence, Union

import torch
from torch import nn

import tensor_parallel.cross_device_ops as cross_device_ops
from tensor_parallel.communications import (
    AllGather,
    AllReduce,
    DistributedAllGather,
    DistributedAllReduce,
    NCCLAllGather,
    NCCLAllReduce,
)
from tensor_parallel.state_actions import LegacyStateAction, Split, StateAction
from tensor_parallel.utils import check_lora

logger = logging.getLogger(__file__)

Arg = Union[int, str]
Action = Callable[[Any, int], Any]  # actions describe what to do with tensors
MappedActions = Dict[Arg, Action]

StateRules = Dict[re.Pattern, StateAction]  # state rules are pattern-matched actions on module state dict
ModuleRules = Dict[re.Pattern, MappedActions]  # module rules are pattern-matched actions on inputs/outputs/attrs


TENSOR_PARALLEL_USE_NATIVE = bool(os.environ.get("TENSOR_PARALLEL_USE_NATIVE"))


@dataclasses.dataclass
class Config:
    state_rules: Dict[str, StateAction]
    input_rules: Dict[str, MappedActions]
    output_rules: Dict[str, MappedActions]
    attr_rules: Dict[str, MappedActions]

    def __init__(
        self, state_rules: StateRules, input_rules: ModuleRules, output_rules: ModuleRules, attr_rules: ModuleRules
    ):
        for pattern, state_action in state_rules.items():
            state_rules[pattern] = convert_legacy_state_action(state_action)

        all_rules = [state_rules, input_rules, output_rules, attr_rules]
        for i, rule_set in enumerate(all_rules):
            all_rules[i] = dict((re.compile(pattern), actions) for pattern, actions in rule_set.items())
        self.state_rules, self.input_rules, self.output_rules, self.attr_rules = all_rules

    def create_collective_ops(self, devices: Sequence[torch.device], distributed: bool = True):
        """
        Return a copy of config with thread-parallel collective operations, such as AllGather and AllReduce

        :note: this function should be called during TensorParallel init, before making shards
        """
        return dataclasses.replace(
            self,
            input_rules=create_collective_ops(self.input_rules, devices, distributed),
            output_rules=create_collective_ops(self.output_rules, devices, distributed),
        )


def convert_legacy_state_action(state_action: Any) -> StateAction:
    if isinstance(state_action, StateAction):
        return state_action
    if callable(state_action):
        logger.warning(f"Using callables in state_rules is deprecated. Please use StateAction instead.")
        return LegacyStateAction(state_action)
    if (
        isinstance(state_action, tuple)
        and len(state_action) == 2
        and callable(state_action[0])
        and callable(state_action[1])
    ):
        logger.warning(f"Using tuples of callables in state_rules is deprecated. Please use StateAction instead.")
        return LegacyStateAction(state_action[0], state_action[1])

    raise Exception(f"Can't convert {state_action} of type {type(state_action)} to StateAction")


def create_collective_ops(rules: dict, devices: Sequence[torch.device], distributed: bool = True):
    """Initialize collective thread-parallel operations from config rules"""
    world_size = len(devices)
    all_cuda = all(device.type == "cuda" for device in devices)
    unique_output_transforms = {op for output_actions in rules.values() for op in output_actions.values()}
    transform_map = {}
    if torch.distributed.is_initialized() and distributed:
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


def add_lora_rules(model: nn.Module, config: Config) -> Config:
    lora_state_rules = {}
    lora_input_rules = {}
    lora_output_rules = {}
    for name, module in model.named_modules():
        if check_lora(module=module):
            for pattern, action in config.state_rules.items():
                if pattern.search(name + ".weight") is not None:
                    if isinstance(action, Split):
                        if action.dim == 0:
                            lora_state_rules[re.compile(rf"^{name}.lora_B")] = action
                        elif action.dim == 1:
                            lora_input_rules[re.compile(rf"^{name}.lora_A.*.")] = {0: "gather -1"}
                            lora_output_rules[re.compile(rf"^{name}.lora_A.*.")] = {0: "scale"}
                        else:
                            raise Exception(
                                "Expected dim in [0, 1]. Don't know what to do with LoRA linear split along dim {i}"
                            )
                    else:
                        raise Exception(
                            f"Can't apply action {action} since it uses LoRA. The only actions with LoRA support are Split and it's variations."
                        )

    config.state_rules.update(lora_state_rules)
    config.input_rules.update(lora_input_rules)
    config.output_rules.update(lora_output_rules)
    return config
