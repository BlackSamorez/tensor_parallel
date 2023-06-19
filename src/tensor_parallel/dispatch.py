import re
from contextlib import contextmanager
from itertools import chain
from typing import Union

import torch

from tensor_parallel.pretrained_model import TensorParallelPreTrainedModel
from tensor_parallel.sharding import Sharded
from tensor_parallel.tensor_parallel import Config, TensorParallel
from tensor_parallel.utils import find_tied_weight_aliases


@contextmanager
def save_tensor_parallel(model: Union[TensorParallel, TensorParallelPreTrainedModel, Sharded]):
    """Enables state_dict reconstruction for tensor_parallel models.
    With it '.state_dict()' produces a state dict that can be loaded into an underlying model.
    Example:
    ```python
    model = <some model>
    model_tp = tensor_parallel(model)
    with save_tensor_parallel(model_tp):
        model.load_state_dict(model_tp.state_dict()) # state dicts match
    ```

    Args:
        model (Union[TensorParallel, TensorParallelPreTrainedModel, Sharded]): tensor_parallel model
    """
    model.preserve_shards_when_saving = False
    try:
        yield
    finally:
        model.preserve_shards_when_saving = True


def infer_sharded_data_device_id(name: str):
    if name.find("_sanity_check_params.") != -1:
        shard_id_start = name.find("_sanity_check_params.") + len("_sanity_check_params.")
    elif name.find("module_shards.") != -1:
        shard_id_start = name.find("module_shards.") + len("module_shards.")
    elif name.find("flat_shards.") != -1:
        shard_id_start = name.find("flat_shards.") + len("flat_shards.")
    else:
        raise KeyError(
            f"Can't decide where to put {name}. Are you sure you passed a tensor_parallel parameter was passed?"
        )

    shard_id_end = name.find(".", shard_id_start)
    return int(name[shard_id_start:shard_id_end]) if shard_id_end > 0 else int(name[shard_id_start:])


def infer_sharded_device_map(tp_model: Union[TensorParallel, TensorParallelPreTrainedModel]) -> dict:
    """Creates a mapping describing which device each tensor parallel model parameter should be put on.

    Args:
        tp_model (Union[TensorParallel, TensorParallelPreTrainedModel]): tensor parallel model

    Returns:
        dict: parameter device mapping
    """
    id_to_device = {}
    for name, param in tp_model.named_parameters():
        id_to_device[id(param)] = tp_model.devices[infer_sharded_data_device_id(name)]
    for name, buffer in tp_model.named_buffers():
        id_to_device[id(buffer)] = tp_model.devices[infer_sharded_data_device_id(name)]
    id_to_aliases = find_tied_weight_aliases(tp_model)

    device_map = {}
    for idx, aliases in id_to_aliases.items():
        for alias in aliases:
            device_map[alias] = id_to_device[idx]

    return device_map


def convert_state_dict(
    input_state_dict, tensor_parallel_config: Config, world_size: int, for_pretrained: bool = False
) -> dict:
    """Creates a state_dict to be loaded into a tensor parallel model from a state_dict of a base model.
    WARNING: this function doesn't properly work with tied weights. You'll probably need to fix the resulting state_dict by hand.

    Args:
        input_state_dict (_type_): state_dict to be converted
        tensor_parallel_config (Config): tensor parallel config for the base model
        world_size (int): number of devices of the tensor parallel model
        for_pretrained (bool, optional): Whether to create a state dict to be loaded into TensorParallelModel of PreTrainedTensorParallelModel. Defaults to False.

    Returns:
        dict: _description_
    """
    output_state_dict = {}
    for i in range(world_size):
        output_state_dict.update(
            convert_state_dict_single_shard(input_state_dict, tensor_parallel_config, world_size, i)
        )
        output_state_dict[f"_sanity_check_params.{i}"] = torch.empty(0, device="cpu")
    if for_pretrained:
        name_replacements = {name: "wrapped_model." + name for name in output_state_dict.keys()}
        for old_name, new_name in name_replacements.items():
            output_state_dict[new_name] = output_state_dict.pop(old_name)
    return output_state_dict


def convert_data(input_state_dict, output_state_dict, tensor_parallel_config: Config, world_size: int, rank: int):
    for name, state in input_state_dict.items():
        for pattern, action in tensor_parallel_config.state_rules.items():
            if pattern.search(name) is not None:
                output_state_dict[name] = action(state, rank=rank)
                break
        else:
            output_state_dict[name] = input_state_dict[name]  # copy source parameter as is


def convert_names(state_dict, tensor_parallel_config: Config):
    patterns = tuple(
        regex.pattern
        for regex in chain(tensor_parallel_config.input_rules.keys(), tensor_parallel_config.output_rules.keys())
    )
    patterns = set(pattern[:-1] + "\." if pattern.endswith("$") else pattern for pattern in patterns)
    patterns = [re.compile(pattern) for pattern in patterns]

    name_replacements = {name: name for name in state_dict.keys()}
    for pattern in patterns:
        for initial_name, old_name in name_replacements.items():
            match = pattern.search(old_name)
            if match is not None:
                end_pos = match.span()[1]
                new_name = old_name[:end_pos] + "tp_wrapped_module." + old_name[end_pos:]
                name_replacements[initial_name] = new_name

    for initial_name, final_name in name_replacements.items():
        state_dict[final_name] = state_dict.pop(initial_name)


def prefix_names_with_shard_id(state_dict, rank: int):
    name_replacements = {name: f"module_shards.{rank}." + name for name in state_dict.keys()}
    for old_name, new_name in name_replacements.items():
        state_dict[new_name] = state_dict.pop(old_name)


def convert_state_dict_single_shard(input_state_dict, tensor_parallel_config: Config, world_size: int, rank: int):
    output_state_dict = {}
    convert_data(input_state_dict, output_state_dict, tensor_parallel_config, world_size, rank)
    convert_names(output_state_dict, tensor_parallel_config)
    prefix_names_with_shard_id(output_state_dict, rank)
    return output_state_dict
