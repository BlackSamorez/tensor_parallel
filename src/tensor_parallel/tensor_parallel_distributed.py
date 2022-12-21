import logging
from typing import Optional

import torch
from torch import nn

from tensor_parallel.slicer_wrapper import Config
from tensor_parallel.tensor_parallel_pretrained_model import find_predefined_tensor_parallel_config

logger = logging.getLogger(__file__)


def get_distributed_shard(module: nn.Module, device: torch.device, config: Optional[Config]):
    if config is None:
        config = Config.get_default_config(module)
        logger.info("Using automatic config: sharding individual linear/conv/emb layers")

    config_with_ops = config.create_collective_ops([torch.device("cpu")] * torch.distributed.get_world_size())
    # ^-- creates a copy of comfig with collective op instances, such as AllReduce and AllGather

    return config.make_shard(
        module,
        device,
        config_with_ops,
        rank=torch.distributed.get_rank(),
        world_size=torch.distributed.get_world_size(),
    )
