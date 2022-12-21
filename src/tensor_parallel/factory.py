from typing import Optional

import torch
from torch import nn
from transformers import PreTrainedModel

from tensor_parallel.slicer_wrapper import Config
from tensor_parallel.tensor_parallel import TensorParallel
from tensor_parallel.tensor_parallel_pretrained_model import TensorParallelPreTrainedModel


def tensor_parallel(module: nn.Module, *args, config: Optional[Config] = None, **kwargs) -> nn.Module:
    if isinstance(module, PreTrainedModel):
        return TensorParallelPreTrainedModel(module, *args, config=config, **kwargs)
    else:
        return TensorParallel(module, *args, config=config, **kwargs)
    # TODO: here be torchrun slices
