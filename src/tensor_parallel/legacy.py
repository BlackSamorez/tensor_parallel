import logging
from typing import Collection, Optional, Sequence, Tuple, Union

import torch
import torch.distributed
from torch import nn
from transformers import PreTrainedModel

from tensor_parallel.config import Config
from tensor_parallel.pretrained_model import TensorParallelPreTrainedModel
from tensor_parallel.shard import make_distributed_shard
from tensor_parallel.tensor_parallel import TensorParallel

logger = logging.getLogger(__file__)


def tensor_parallel(
    module: nn.Module,
    device_ids: Optional[Sequence[Union[torch.device, str]]] = None,
    tensor_parallel_config: Optional[Config] = None,
    distributed: Optional[bool] = None,
    sharded: Optional[bool] = None,
    sharded_param_names: Optional[Collection[str]] = None,
    **kwargs,
) -> nn.Module:
    """
    !!! This function is deprecated and will be deleted in subsequent releases!!!

    Wrap an existing PyTorch module with tensor parallelism. Return equivalent tensor-parallel module.

    :example:

    >>> import torch, transformers
    >>> import tensor_parallel as tp
    >>> model = transformers.AutoModel.from_pretrained("t5-11b")
    >>> model = tp.tensor_parallel(model, device_ids=['cuda:0', 'cuda:1'])
    >>> outputs_as_usual = model(**inputs_as_usual)  # backprop also works!

    :param module: original PyTorch module. We recommend storing input module on CPU to minimize GPU memory
    :param device_ids: model will be split among this list of devices (e.g. GPUs), default = all available CUDA devices
    :param tensor_parallel_config: custom tensor_parallel.Config to describe how the model is parallelized. defaults to auto config
    :param distributed: if True, use torch.distributed instead of threading. Assumes that we is running in torchrun
       defaults to True if torch.distributed.is_initialized, else False
    :param sharded: if True, any non-tensor-parallel parameters (e.g. layernorm weight) will still be sharded,
       and manually re-assembled for each forward. This is equivalent to pytorch FullyShardedDataParallel
    :param sharded_param_names: if sharded=True, this is a list of all parameter names (strings) that ZeRO-3 applies to;
       by default, ZeRO-3 applies to all parameters that are not split with tensor parallelism.
    :note: the default sharded_param_names are formed of parameters that are equal between shards after TP is applied
    :param kwargs: additional keyword arguments passed to TensorParallel init

    """
    logger.warning(
        f"`tensor_parallel` is deprecated. Please use `TensorParallel`, `TensorParallelPreTrainedModel` or `make_distributed_shard`  accordingly"
    )

    distributed = distributed if distributed is not None else torch.distributed.is_initialized()

    if distributed:
        if device_ids is None:
            device_ids = [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
        assert len(device_ids) == 1, "if distributed=True, please specify a single (current) device"
        assert not sharded, "distributed + sharded mode is not implemented, please keep one"

        return make_distributed_shard(module, device=torch.device(device_ids[0]), **kwargs)
    else:
        if isinstance(module, PreTrainedModel):
            return TensorParallelPreTrainedModel(
                module,
                device_ids=device_ids,
                tensor_parallel_config=tensor_parallel_config,
                distributed=distributed,
                use_zero3=sharded,
                replicated_param_names=sharded_param_names,
                **kwargs,
            )
        else:
            return TensorParallel(
                module,
                device_ids=device_ids,
                tensor_parallel_config=tensor_parallel_config,
                distributed=distributed,
                use_zero3=sharded,
                replicated_param_names=sharded_param_names,
                **kwargs,
            )


class Sharded(nn.Module):
    def __new__(
        cls,
        module: Tuple[TensorParallel, TensorParallelPreTrainedModel],
        sharded_param_names: Optional[Collection[str]] = None,
    ):
        logger.warning(f"`Sharded` is deprecated. Please use `.use_zero3()` method")
        module.use_zero3(sharded_param_names)
        return module
