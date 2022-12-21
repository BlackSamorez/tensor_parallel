from typing import Optional, Sequence

import torch
from torch import nn
from transformers import PreTrainedModel

from tensor_parallel.slicer_wrapper import Config
from tensor_parallel.tensor_parallel import TensorParallel
from tensor_parallel.tensor_parallel_distributed import get_distributed_shard
from tensor_parallel.tensor_parallel_pretrained_model import TensorParallelPreTrainedModel


def tensor_parallel(
    module: nn.Module, device_ids: Optional[Sequence[torch.device]] = None, config: Optional[Config] = None, **kwargs
) -> nn.Module:
    if torch.distributed.is_initialized():
        return get_distributed_shard(module, device=torch.device(device_ids[0]), config=config, **kwargs)
    else:
        if isinstance(module, PreTrainedModel):
            return TensorParallelPreTrainedModel(module, device_ids=device_ids, config=config, **kwargs)
        else:
            return TensorParallel(module, device_ids=device_ids, config=config, **kwargs)
