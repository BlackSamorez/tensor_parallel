"""
Utility functions for training original model parameters
"""
import functools
import logging
import os
from typing import Collection, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from tensor_parallel.config import TENSOR_PARALLEL_USE_NATIVE
from tensor_parallel.cross_device_ops import all_gather

logger = logging.getLogger(__file__)


class Sharded(nn.Module):
    def __init__(
        self,
        module,
        world_size: int,
        replicated_param_names: Optional[Collection[str]] = None,
    ):
        """
        Wrap a TensorParallel module and partition the specified param_names between shards so that every parameter
        value is only present on a single device.
        """
        super().__init__()
        self.preserve_shards_when_saving: bool = True

        module_shards = module.module_shards
        if len(module_shards) == 1:
            return
        if replicated_param_names is None:
            if any([p.device.type == "meta" for p in module.parameters()]):
                raise RuntimeError(
                    "Trying to shard a model containing 'meta' parameters. Please set `sharded=False` during model creation and call `.apply_sharding()` only after dispatch"
                )

            all_param_names = set(name for name, _ in module_shards[0].named_parameters())
            replicated_param_names = all_param_names - module.modified_parameters_names

        self.sharded_param_names = replicated_param_names = tuple(replicated_param_names)
        if not replicated_param_names:
            logger.warning("Did not find any parameters to shard")
            return

        self.world_size = world_size = world_size if world_size is not None else len(module_shards)
        sharded_param_names_set = set(replicated_param_names)
        all_param_shapes = {
            name: param.shape for name, param in module_shards[0].named_parameters() if name in sharded_param_names_set
        }
        self._sharded_param_shapes = [all_param_shapes[name] for name in replicated_param_names]
        flat_shards, shard_sizes_with_pad = _make_flat_shards(replicated_param_names, module_shards, world_size)
        self.flat_shards = nn.ParameterList(list(map(nn.Parameter, flat_shards)))
        self._shard_sizes_with_pad = shard_sizes_with_pad

        # prepare a list of all module-parameter pairs affected by sharding
        self.param_occurences_by_rank = [_find_all_occurences(shard, replicated_param_names) for shard in module_shards]

        # remove original parameters
        for param_occurences in self.param_occurences_by_rank:
            for occurences in param_occurences:
                for submodule, param_name in occurences:
                    submodule._parameters.pop(param_name, None)
                    setattr(submodule, param_name, None)
        self._last_versions = None  # to be updated during first forward

    def synchronize_weights(self, all_cuda: bool):
        shard_versions = tuple(flat_shard._version for flat_shard in self.flat_shards)
        if shard_versions == self._last_versions:
            logger.debug("Using previously gathered parameters")
            return  # parameters did not change since last all-gather; keep old versions

        logger.debug("Gathering sharded parameters")
        gathered_shards = all_gather(list(self.flat_shards), all_cuda=all_cuda)
        for flat_shards, param_occurences in zip(gathered_shards, self.param_occurences_by_rank):
            combined_params = _combine_shards(flat_shards, self._shard_sizes_with_pad, self._sharded_param_shapes)
            assert len(combined_params) == len(param_occurences)
            for new_value, occurences in zip(combined_params, param_occurences):
                for submodule, param_name in occurences:
                    setattr(submodule, param_name, new_value)
        self._last_versions = tuple(flat_shard._version for flat_shard in self.flat_shards)

    def state_dict(self, *args, **kwargs):
        if self.preserve_shards_when_saving:
            return super().state_dict(*args, **kwargs)
        else:
            return kwargs["destination"]


@torch.no_grad()
def _extract_param_shards(
    model: nn.Module, sharded_param_names: Collection[str], *, rank: int, world_size: int
) -> Sequence[torch.Tensor]:
    """Find the specified params and return param_shard tensors in the same order as in sharded_param_names"""
    shards = {}
    for full_param_name, param in model.named_parameters(recurse=True):
        if full_param_name in sharded_param_names:
            logger.debug(f"Sharding {full_param_name}")
            shards[full_param_name] = torch.tensor_split(param.data.flatten(), world_size)[rank]
        else:
            logger.debug(f"Not sharding {full_param_name}")
    return [shards[name] for name in sharded_param_names]


@torch.no_grad()
def _find_all_occurences(model: nn.Module, param_names: Sequence[str]) -> Sequence[List[Tuple[nn.Module, str]]]:
    """Find all occurences of the specified params, including tied / shared params"""
    param_occurences = {name: [] for name in param_names}
    alias_param_names_by_id: Dict[int, str] = dict()

    # find alternative param_names from tied weights
    for module_name, submodule in model.named_modules():
        for param_name, param in submodule.named_parameters(recurse=False):
            param_id = hash((param.data.data_ptr(), param.data.storage_offset(), param.stride(), param.shape))
            full_param_name = ".".join((module_name, param_name))
            if full_param_name in param_occurences:
                alias_param_names_by_id[param_id] = full_param_name

    # find alternative param_names from tied weights
    for module_name, submodule in model.named_modules():
        for param_name, param in submodule.named_parameters(recurse=False):
            param_id = hash((param.data.data_ptr(), param.data.storage_offset(), param.stride(), param.shape))
            full_param_name = alias_param_names_by_id.get(hash(param_id), ".".join((module_name, param_name)))
            if full_param_name in param_occurences:
                param_occurences[full_param_name].append((submodule, param_name))

    for name in param_names:
        assert param_occurences[name], f"could not find param {name} in model"
    return tuple(param_occurences[name] for name in param_names)


@torch.no_grad()
def _make_flat_shards(
    sharded_param_names: Sequence[str], module_shards: Sequence[nn.Module], world_size: int
) -> Tuple[List[torch.Tensor], List[List[int]]]:
    """Create 1d buffers containing all parameter shards for each rank, return buffers and the original shard sizes"""
    extracted_shards = {
        i: _extract_param_shards(shard, sharded_param_names, rank=i, world_size=world_size)
        for i, shard in enumerate(module_shards)
    }
    shard_sizes = {rank: [shard.numel() for shard in shards] for rank, shards in extracted_shards.items()}
    shard_totals = {rank: sum(numel) for rank, numel in shard_sizes.items()}
    max_size = max(shard_totals.values())
    paddings = {rank: max_size - shard_size for rank, shard_size in shard_totals.items()}
    shard_sizes_with_pad = [shard_sizes[i] + [paddings[i]] for i in range(len(module_shards))]

    padding_tensors = {
        rank: torch.zeros(paddings[rank], device=shards[0].device, dtype=shards[0].dtype)
        for rank, shards in extracted_shards.items()
    }
    flat_shards = [torch.cat(extracted_shards[i] + [padding_tensors[i]]) for i in range(len(module_shards))]
    assert len(set(len(flat_shard) for flat_shard in flat_shards)) == 1
    return flat_shards, shard_sizes_with_pad


@functools.lru_cache()
def maybe_script(fn: callable) -> callable:
    """Apply torch.jit.script to function unless one is using TPU. TPU does not support torch.jit.script."""
    using_tpu = bool(os.environ.get("TPU_NAME"))
    # this is a reserved variable that must be set to TPU address (e.g. grpc://11.22.33.44:1337) for TPU to function
    should_script = not TENSOR_PARALLEL_USE_NATIVE and not os.getenv("TENSOR_PARALLEL_NO_SCRIPT") and not using_tpu
    return torch.jit.script(fn) if should_script else fn


@maybe_script
def _combine_shards(
    gathered_flat_shards: torch.Tensor, shard_sizes_with_pad: List[List[int]], tensor_shapes: List[List[int]]
) -> List[torch.Tensor]:
    """Split each flat shard into individual tensors, combine tensors across shards and return un-sharded tensors"""
    shard_tensors: List[List[torch.Tensor]] = []
    for shard_i in range(len(gathered_flat_shards)):
        shard_tensors.append(torch.split_with_sizes(gathered_flat_shards[shard_i], shard_sizes_with_pad[shard_i]))
    output_tensors: List[torch.Tensor] = []
    for tensor_i in range(len(tensor_shapes)):
        shards = [shard_tensors[shard_j][tensor_i] for shard_j in range(len(gathered_flat_shards))]
        output_tensors.append(torch.cat(shards, dim=0).view(tensor_shapes[tensor_i]))
    return output_tensors
