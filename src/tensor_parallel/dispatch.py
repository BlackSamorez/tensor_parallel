import json
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, Union

import torch
from accelerate import load_checkpoint_in_model

from tensor_parallel.pretrained_model import TensorParallelPreTrainedModel
from tensor_parallel.sharding import Sharded
from tensor_parallel.tensor_parallel import TensorParallel, check_device_ids


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
    else:
        raise KeyError(
            "Can't decide where to put {name} in a sharded model state dict. Are you sure it's a sharded dict?"
        )

    shard_id_end = name.find(".", shard_id_start)
    return int(name[shard_id_start:shard_id_end]) if shard_id_end > 0 else int(name[shard_id_start:])


def infer_sharded_device_map(tp_model):
    device_map = {}
    for name, _ in tp_model.named_parameters():
        device_map[name] = tp_model.devices[infer_sharded_data_device_id(name)]
    for name, _ in tp_model.named_buffers():
        device_map[name] = tp_model.devices[infer_sharded_data_device_id(name)]
    return device_map
