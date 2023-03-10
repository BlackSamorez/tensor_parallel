import json
import os
from typing import Any, Dict, Optional, Sequence, Union

import torch
from accelerate import load_checkpoint_in_model

from tensor_parallel.pretrained_model import TensorParallelPreTrainedModel
from tensor_parallel.tensor_parallel import TensorParallel, check_device_ids


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
