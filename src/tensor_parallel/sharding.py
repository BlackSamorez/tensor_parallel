"""
Utility functions for training original model parameters
"""
import functools
import logging
import os
from collections import OrderedDict
from operator import attrgetter
from typing import Collection, Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn

from tensor_parallel.config import TENSOR_PARALLEL_USE_NATIVE
from tensor_parallel.cross_device_ops import all_gather
from tensor_parallel.tensor_parallel import TensorParallel

logger = logging.getLogger(__file__)


class Sharded(nn.ModuleList):
    def __init__(
        self,
        module: TensorParallel,
        sharded_param_names: Optional[Collection[str]] = None,
        ranks: Optional[Sequence[int]] = None,
        world_size: Optional[int] = None,
    ):
        """
        Wrap a TensorParallel module and partition the specified param_names between shards so that every parameter
        value is only present on a single device.
        """
        super().__init__()
        assert isinstance(module, TensorParallel), "expected TensorParallel module"
        self.module = module

        module_shards = module.module_shards
        if len(module_shards) == 1:
            return
        if sharded_param_names is None:
            if any([p.device.type == "meta" for p in module.parameters()]):
                raise RuntimeError(
                    "Trying to shard a model containing data on 'meta' device without providing 'sharded_param_names'. Consider sharding a model after loading the data for automatic sharding."
                )

            sharded_param_names = find_replicated_parameters(*module_shards, only_trainable=True)

        self.sharded_param_names = sharded_param_names = tuple(sharded_param_names)
        if not sharded_param_names:
            logger.warning("Did not find any parameters to shard")
            return

        self.ranks = ranks = tuple(ranks if ranks is not None else range(len(module_shards)))
        assert ranks == tuple(sorted(ranks)), "ranks (and module shards) must be ordered from lowest to highest rank"
        self.world_size = world_size = world_size if world_size is not None else len(module_shards)
        sharded_param_names_set = set(sharded_param_names)
        all_param_shapes = {
            name: param.shape for name, param in module_shards[0].named_parameters() if name in sharded_param_names_set
        }
        self._sharded_param_shapes = [all_param_shapes[name] for name in sharded_param_names]
        flat_shards, shard_sizes_with_pad = _make_flat_shards(sharded_param_names, module_shards, ranks, world_size)
        self.flat_shards = nn.ParameterList(list(map(nn.Parameter, flat_shards)))
        self._shard_sizes_with_pad = shard_sizes_with_pad

        # prepare a list of all module-parameter pairs affected by sharding
        self.param_occurences_by_rank = [_find_all_occurences(shard, sharded_param_names) for shard in module_shards]

        # remove original parameters
        for param_occurences in self.param_occurences_by_rank:
            for occurences in param_occurences:
                for submodule, param_name in occurences:
                    submodule._parameters.pop(param_name, None)
                    setattr(submodule, param_name, None)
        self._last_versions = None  # to be updated during first forward

    @property
    def devices(self):
        return self.module.devices

    @property
    def tensor_parallel_config(self):
        return self.module.tensor_parallel_config

    @property
    def preserve_shards_when_saving(self):
        return self.module.preserve_shards_when_saving

    @preserve_shards_when_saving.setter
    def preserve_shards_when_saving(self, value):
        self.module.preserve_shards_when_saving = value

    def forward(self, *args, **kwargs):
        if len(self.module.module_shards) > 1 and len(self.sharded_param_names) > 0:
            self._maybe_fill_sharded_params()
        return self.module(*args, **kwargs)

    def _maybe_fill_sharded_params(self):
        shard_versions = tuple(flat_shard._version for flat_shard in self.flat_shards)
        if shard_versions == self._last_versions:
            logger.debug("Using previously gathered parameters")
            return  # parameters did not change since last all-gather; keep old versions

        logger.debug("Gathering sharded parameters")
        gathered_shards = all_gather(list(self.flat_shards), all_cuda=self.module.all_cuda)
        for rank, flat_shards, param_occurences in zip(self.ranks, gathered_shards, self.param_occurences_by_rank):
            combined_params = _combine_shards(flat_shards, self._shard_sizes_with_pad, self._sharded_param_shapes)
            assert len(combined_params) == len(param_occurences)
            for new_value, occurences in zip(combined_params, param_occurences):
                for submodule, param_name in occurences:
                    setattr(submodule, param_name, new_value)
        self._last_versions = tuple(flat_shard._version for flat_shard in self.flat_shards)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if self.module.preserve_shards_when_saving:
            return super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        self._maybe_fill_sharded_params()

        for name in self.sharded_param_names:
            for shard in self.module.module_shards:
                try:
                    destination[prefix + name] = attrgetter(name)(shard)
                    break
                except KeyError:
                    pass

        state_dict = self.module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return state_dict

    @property
    def module_shards(self):
        return self.module.module_shards


@torch.no_grad()
def find_replicated_parameters(*module_shards: nn.Module, only_trainable: bool) -> Set[str]:
    """
    Detects parameters that are replicated across model shards, return a set parameter names
    Parameters are replicated if they have the same value (and name) across all shards
    """
    assert len(module_shards) > 1, "please specify several module shards as *args"
    named_params_by_shard = [dict(shard.named_parameters()) for shard in module_shards]
    param_names = set(named_params_by_shard[0].keys())

    replicated_params = set()
    for param_name in param_names:
        shard0_value = named_params_by_shard[0][param_name]
        for i in range(len(module_shards)):
            shard_i_value = named_params_by_shard[i].get(param_name)
            if shard_i_value is None:
                logger.warning(f"Parameter {param_name} not found in module shard {i}")
                break
            if only_trainable and not shard_i_value.requires_grad:
                break
            if shard0_value.shape != shard_i_value.shape or shard0_value.dtype != shard_i_value.dtype:
                break
            if torch.any(torch.ne(shard0_value.cpu(), shard_i_value.cpu())):
                break
        else:
            replicated_params.add(param_name)
    return replicated_params


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
    sharded_param_names: Sequence[str], module_shards: Sequence[nn.Module], ranks: Sequence[int], world_size: int
) -> Tuple[List[torch.Tensor], List[List[int]]]:
    """Create 1d buffers containing all parameter shards for each rank, return buffers and the original shard sizes"""
    extracted_shards = {
        rank: _extract_param_shards(shard, sharded_param_names, rank=rank, world_size=world_size)
        for rank, shard in zip(ranks, module_shards)
    }
    shard_sizes = {rank: [shard.numel() for shard in shards] for rank, shards in extracted_shards.items()}
    shard_totals = {rank: sum(numel) for rank, numel in shard_sizes.items()}
    max_size = max(shard_totals.values())
    paddings = {rank: max_size - shard_size for rank, shard_size in shard_totals.items()}
    shard_sizes_with_pad = [shard_sizes[rank] + [paddings[rank]] for rank in ranks]

    padding_tensors = {
        rank: torch.zeros(paddings[rank], device=shards[0].device, dtype=shards[0].dtype)
        for rank, shards in extracted_shards.items()
    }
    flat_shards = [torch.cat(extracted_shards[rank] + [padding_tensors[rank]]) for rank in ranks]
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
