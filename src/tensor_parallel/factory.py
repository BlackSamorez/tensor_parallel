import logging
from typing import Optional, Sequence

import torch
import torch.distributed
from torch import nn
from transformers import PreTrainedModel

from tensor_parallel.pretrained_model import TensorParallelPreTrainedModel
from tensor_parallel.slicer_wrapper import Config
from tensor_parallel.tensor_parallel import TensorParallel

logger = logging.getLogger(__file__)


def tensor_parallel(
    module: nn.Module,
    device_ids: Optional[Sequence[torch.device]] = None,
    config: Optional[Config] = None,
    distributed: Optional[bool] = None,
    **kwargs
) -> nn.Module:
    distributed = distributed if distributed is not None else torch.distributed.is_initialized()
    if distributed:
        if device_ids is None:
            device_ids = [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
        assert len(device_ids) == 1, "if distributed=True, please specify a single (current) device"
        if config is None:
            config = Config.get_default_config(module, device_ids=range(torch.distributed.get_world_size()))
            logger.info("Using automatic config: sharding individual linear/conv/emb layers")

        return config.make_distributed_shard(module, device=torch.device(device_ids[0]), **kwargs)
    else:
        if isinstance(module, PreTrainedModel):
            return TensorParallelPreTrainedModel(module, device_ids=device_ids, config=config, **kwargs)
        else:
            return TensorParallel(module, device_ids=device_ids, config=config, **kwargs)
