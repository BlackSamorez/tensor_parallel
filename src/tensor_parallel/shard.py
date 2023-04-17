import logging
from copy import deepcopy
from itertools import chain
from typing import Dict, Optional

import torch
from torch import nn

from tensor_parallel.autoconfig import get_default_config
from tensor_parallel.config import Config, MappedActions, ModuleRules
from tensor_parallel.wrapper import TensorParallelWrapper

logger = logging.getLogger(__file__)


def make_shard(module: nn.Module, device: torch.device, config: Config, *, rank: int, world_size: int) -> nn.Module:
    """Apply a tensor-parallel config to a high-level module, return module shard for a given rank and device"""
    assert (
        len(list(module.children())) != 0
    ), "Please ensure module is a container (e.g. Sequential), not a single layer"
    source_tensors = dict(chain(module.named_parameters(), module.named_buffers()))
    substitutes = {
        id(x): nn.Parameter(
            torch.empty(
                0,
                dtype=x.dtype,
                device=device if x.device.type != "meta" else x.device,
                requires_grad=x.requires_grad,
            ),
            x.requires_grad,
        )
        if isinstance(x, nn.Parameter)
        else torch.empty(
            0, dtype=x.dtype, device=device if x.device.type != "meta" else x.device, requires_grad=x.requires_grad
        )
        for x in source_tensors.values()
    }
    shard = deepcopy(module, memo=substitutes)
    # ^-- note: the memo=... above will replace all parameters and buffers with empty tensors
    del module, substitutes

    # convert parameters and buffers
    process_state_(shard, source_tensors, config, rank=rank, world_size=world_size)
    del source_tensors

    # convert or wrap intermediate modules
    unique_wrappers = {}
    with torch.no_grad():
        for name, submodule in shard.named_modules():
            if submodule in unique_wrappers:
                continue  # wrap a module at most once
            maybe_wrapped = _maybe_wrap_submodule(config, name, submodule, rank=rank, world_size=world_size)
            if maybe_wrapped:  # apply the same wrapper to all occurrences of the given module
                unique_wrappers[submodule] = maybe_wrapped

    for parent in list(shard.modules()):
        for child_name, child in list(parent.named_children()):
            if child in unique_wrappers:
                setattr(parent, child_name, unique_wrappers[child])

    # automatically fixes certain submodule attributes such that
    # it's not necessary to specify in a config
    fix_general_attributes(shard)

    return unique_wrappers.get(shard, shard)  # wrap the root module if needed


def make_distributed_shard(module: nn.Module, device: torch.device, config: Optional[Config] = None):
    if config is None:
        config = get_default_config(module, device_ids=range(torch.distributed.get_world_size()))
        logger.info("Using automatic config: sharding individual linear/conv/emb layers")

    config_with_ops = config.create_collective_ops([torch.device("cpu")] * torch.distributed.get_world_size())
    return make_shard(
        module,
        device,
        config_with_ops,
        rank=torch.distributed.get_rank(),
        world_size=torch.distributed.get_world_size(),
    )


def _maybe_wrap_submodule(config: Config, name: str, module: nn.Module, *, rank: int, world_size: int) -> nn.Module:
    """
    Apply the tensor parallelism config to the specified module, return wrapped module

    :note: this function does NOT apply state dict changes (state_rules), these should be applied separately!
    :note: this function will modify the original module in-place!
    """
    attr_actions = find_matching_actions("attr", name, config.attr_rules)
    input_actions = find_matching_actions("input", name, config.input_rules)
    output_actions = find_matching_actions("output", name, config.output_rules)
    if attr_actions:
        process_attrs_(module, attr_actions, rank=rank, world_size=world_size)
    if input_actions or output_actions:
        module = TensorParallelWrapper(module, input_actions, output_actions, rank=rank, world_size=world_size)
    return module


def find_matching_actions(action_type: str, name: str, rules: ModuleRules) -> MappedActions:
    found_actions = {}
    for pattern, actions_for_pattern in rules.items():
        if pattern.search(name) is not None:
            for key, action in actions_for_pattern.items():
                if isinstance(key, str) and key.strip().isdigit():
                    key = int(key)  # canonoicalize int keys
                if found_actions.get(key, action) != action:
                    raise ValueError(
                        f"Found conflicting {action_type} rule for module {name}, key {key}:"
                        f" {found_actions[key]} vs {action}"
                    )
                found_actions[key] = action
    return found_actions


@torch.no_grad()
def process_state_(
    sharded_module: nn.Module, source_tensors: Dict[str, torch.Tensor], config: Config, *, rank: int, world_size: int
):
    """
    Initialize sharded_module's parameters and buffers by applying prescribed rules to source_module's buffers
    :param sharded_module: target module that will be modified in-place
    :param source_tensors: original parameters and buffers on a source device
    """
    unused_patterns = set(config.state_rules.keys())
    for name, state in chain(sharded_module.named_parameters(), sharded_module.named_buffers()):
        for pattern, action in config.state_rules.items():
            if pattern.search(name) is not None:
                new_data = action(source_tensors[name], rank=rank)
                unused_patterns.discard(pattern)
                break
        else:
            new_data = source_tensors[name]  # copy source parameter as is

        state.data = new_data.clone().detach().to(state.device).requires_grad_(state.requires_grad)
        # note: .clone is required so that the resulting parameter owns its storage

    if unused_patterns:
        logger.warning(f"The following patterns in state_rules were unused: {[str(p) for p in unused_patterns]}")


def process_attrs_(module: nn.Module, actions: MappedActions, rank: int, world_size: int):
    """Modify module properties in-place"""
    assert not getattr(module, "__tensor_parallel_process_attrs", False), "process_attrs was called more than once"
    for attr, action in actions.items():
        setattr(module, attr, action(getattr(module, attr), rank=rank))
    module.__tensor_parallel_process_attrs = True


@torch.no_grad()
def fix_general_attributes(sharded_module: nn.Module):
    """Fix well-known submodule attributes of a freshly initialized sharded_module.
       For examle fixes nn.Linear in_features and out_features based on a split weight shape.
    Args:
        sharded_module (nn.Module): sharded_module to fix
    """
    for module in sharded_module.modules():
        if isinstance(module, nn.Linear):
            module.in_features = module.weight.shape[1]
            module.out_features = module.weight.shape[0]
