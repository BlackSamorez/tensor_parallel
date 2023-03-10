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


def infer_sharded_device_map(tp_model, devices=None):
    device_map = {}
    for name, _ in tp_model.named_parameters():
        device_map[name] = devices[infer_sharded_data_device_id(name)]
    for name, _ in tp_model.named_buffers():
        device_map[name] = devices[infer_sharded_data_device_id(name)]
    return device_map


def load_and_dispatch_separate_shards(
    model: Union[TensorParallel, TensorParallelPreTrainedModel],
    checkpoint: Union[str, os.PathLike],
    device_ids: Optional[Sequence[Union[torch.device, str]]] = None,
    **kwargs
):
    """Loads a 'tensor_parallel' model saved with 'set_preserve_shards_when_saving(True)' and dispathces it on 'device_ids' for tensor parallel training/inference.
    Is usefult when there is no way to load whole model directly into RAM. Uses 'accelerate' to load shardsa and dispatch the model. **kwargs are passed to 'accelerate.load_checkpoint_in_model'.

    Args:
        model (Union[TensorParallel, TensorParallelPreTrainedModel]): A model to load state dict into.
        checkpoint (Union[str, os.PathLike]): Path to model file or index.json. Passed to
        device_ids (Optional[Sequence[Union[torch.device, str]]], optional): A list of devices to dispatch model to. Length must match number of shards of the model. Defaults to all available GPUs.
    """
    devices = check_device_ids(device_ids)

    load_checkpoint_in_model(
        model, checkpoint=checkpoint, device_map=infer_sharded_device_map(model, devices=devices), **kwargs
    )
    model.set_devices(devices)
