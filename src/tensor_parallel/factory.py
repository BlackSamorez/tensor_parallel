import logging
from typing import Collection, Optional, Sequence, Union

import torch
import torch.distributed
from torch import nn
from transformers import PreTrainedModel

from tensor_parallel.config import Config
from tensor_parallel.pretrained_model import TensorParallelPreTrainedModel
from tensor_parallel.shard import make_distributed_shard
from tensor_parallel.sharding import Sharded
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
    num_trainable_parameters = sum(p.numel() for p in module.parameters() if p.requires_grad)
    distributed = distributed if distributed is not None else torch.distributed.is_initialized()

    if distributed:
        if device_ids is None:
            device_ids = [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
        assert len(device_ids) == 1, "if distributed=True, please specify a single (current) device"
        assert not sharded, "distributed + sharded mode is not implemented, please keep one"

        return make_distributed_shard(module, device=torch.device(device_ids[0]), **kwargs)
    else:
        if isinstance(module, PreTrainedModel):
            module = TensorParallelPreTrainedModel(
                module,
                device_ids=device_ids,
                tensor_parallel_config=tensor_parallel_config,
                distributed=distributed,
                **kwargs,
            )
            module.wrapped_model = _maybe_sharded(
                module.wrapped_model, sharded, num_trainable_parameters, sharded_param_names=sharded_param_names
            )
        else:
            module = TensorParallel(
                module,
                device_ids=device_ids,
                tensor_parallel_config=tensor_parallel_config,
                distributed=distributed,
                **kwargs,
            )
            module = _maybe_sharded(module, sharded, num_trainable_parameters, sharded_param_names=sharded_param_names)

        return module


def _maybe_sharded(
    module: TensorParallel,
    sharded: Optional[bool],
    num_trainable_parameters: int,
    sharded_param_names: Optional[Collection[str]],
    **kwargs,
) -> Union[Sharded, TensorParallel]:
    """Determines if sharding is necessary, returns either Sharded(module) or module itself, if unchanged"""
    determined_automatically = sharded is None
    if sharded is None:
        num_trainable_parameters_after_tp = sum(p.numel() for p in module.parameters() if p.requires_grad)
        assert num_trainable_parameters_after_tp >= num_trainable_parameters
        sharded = num_trainable_parameters_after_tp > num_trainable_parameters
        # use sharding if there are some *trainable* parameter that are replicated on more than one device

    model_is_meta = any([p.device.type == "meta" for p in module.parameters()])
    if sharded and model_is_meta and sharded_param_names is None:
        logger.warning(
            f"Not sharding the model that should be sharded because it has meta tensors which prevent sharding without 'sharded_param_names'. It's recomended to shard a model after loading it's weights."
        )
        sharded = False
    elif sharded and determined_automatically:
        num_extra_parameters = num_trainable_parameters_after_tp - num_trainable_parameters
        replicated_parameters = num_extra_parameters // max(1, len(module.devices) - 1)
        logger.warning(f"Using ZeRO-3 sharding for {replicated_parameters} non tensor-parallel parameters")

    return Sharded(module, sharded_param_names=sharded_param_names, **kwargs) if sharded else module
