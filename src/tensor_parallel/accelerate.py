import json
import os

import torch


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


def save_preserving_shards(model, path: str):
    model.set_preserve_shards_when_saving(True)
    state_dict = model.state_dict()

    index = {}
    reverse_index = {}
    for name, _ in model.named_parameters():
        device_id = infer_sharded_data_device_id(name)
        index[name] = device_id
        if device_id in reverse_index:
            reverse_index[device_id].append(name)
        else:
            reverse_index[device_id] = [name]

    for device_id, shard_data_names in reverse_index.items():
        shard_file_name = f"shard_{device_id}.bin"

        torch.save({name: state_dict[name] for name in shard_data_names}, os.path.join(path, shard_file_name))
        for name in shard_data_names:
            index[name] = shard_file_name

    with open(os.path.join(path, "index.json"), "w") as file:
        json.dump(index, file)
