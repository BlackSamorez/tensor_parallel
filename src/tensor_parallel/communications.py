"""
An interface for thread-parallel collective operations for use in TensorParallel
"""

from __future__ import annotations

import threading
from typing import Any, List, Optional

import torch
from torch.distributed import all_gather, all_reduce

import tensor_parallel.cross_device_ops as cross_device_ops


class CollectiveOpetationBase:
    def __call__(self, x: torch.Tensor, rank: int):
        raise NotImplementedError()


class CollectiveOperation(CollectiveOpetationBase):
    def __init__(self, world_size: int, func: callable, authoritative_rank: int = 0):
        """
        Apply user-defined collective function in a way that is compatible with TensorParallel
        :param func: function(input0, input1, ..., input_worldsize) -> (output0, output1, ..., output_worldsize)
        """
        self.world_size = world_size
        self.func = func
        self.authoritative_rank = authoritative_rank
        self.rank_inputs: List[Any] = [None for _ in range(world_size)]
        self.rank_outputs: List[Any] = [None for _ in range(world_size)]
        self.barrier = threading.Barrier(world_size)

    def __call__(self, x: torch.Tensor, rank: int):
        try:
            self.rank_inputs[rank] = x
            self.barrier.wait()
            if rank == self.authoritative_rank:
                try:
                    result = self.func(*self.rank_inputs)
                    for i in range(self.world_size):
                        self.rank_outputs[i] = (result[i], None)
                except Exception as e:
                    for i in range(self.world_size):
                        self.rank_outputs[i] = (None, e)
            self.barrier.wait()
            result, exception = self.rank_outputs[rank]
            if exception:
                raise exception
            return result
        finally:
            self.rank_inputs[rank] = self.rank_outputs[rank] = None


class NCCLAllReduce(CollectiveOperation):
    def __init__(self, world_size: int):
        super().__init__(world_size, func=cross_device_ops.NCCLAllReduceFunction.apply)


class NCCLAllGather(CollectiveOperation):
    def __init__(self, world_size: int, dim: int):
        super().__init__(world_size, func=cross_device_ops.NCCLAllGatherFunction.apply)
        self.tensor_lengths = [None for _ in range(world_size)]
        self.dim = dim

    def __call__(self, x: torch.Tensor, rank: int):
        # note: all-gather deliberately makes all device-local computations on their ranks, because failing to do so
        # (making one of the ranks operate on other's tensors) may deadlock cuda streams in parallel_apply;
        # if you modify this code and need cross-rank computations, use parallel_apply_simple to avoid stream deadlocks
        try:
            gather_dim = self.dim % x.ndim
            self.tensor_lengths[rank] = x.shape[gather_dim]
            self.barrier.wait()
            tensor_lengths = tuple(self.tensor_lengths)
            max_length = max(tensor_lengths)
            if x.shape[gather_dim] < max_length:
                pad = [0] * (x.ndim * 2)
                pad[2 * (x.ndim - gather_dim - 1) + 1] = max_length - x.shape[gather_dim]
                # to understand the weird indexing (2 * (ndim - dim) ...), see F.pad documentation
                x = torch.nn.functional.pad(x, pad)
            gathered_tensor = super().__call__(x, rank=rank)
            used_padding = any(length != max_length for length in tensor_lengths)

            if not used_padding:
                dim_indices = list(range(1, gathered_tensor.ndim))
                dim_indices.insert(gather_dim, 0)
                concatenated_shape = list(x.shape)
                concatenated_shape[gather_dim] = -1
                return gathered_tensor.permute(dim_indices).reshape(concatenated_shape)
            else:
                # restore original tensor lengths by slicing off padding
                gathered_tensor_parts = []
                for i, tensor_length in enumerate(tensor_lengths):
                    slices_i = [slice(None) for _ in range(gathered_tensor.ndim)]
                    slices_i[0] = i  # select i-th element from gathered tensors
                    slices_i[gather_dim + 1] = slice(0, tensor_length)  # remove padding
                    gathered_tensor_parts.append(gathered_tensor[slices_i])
                return torch.cat(gathered_tensor_parts, dim=self.dim)
        finally:
            self.tensor_lengths[rank] = None


class AllReduce(CollectiveOpetationBase):
    def __init__(
        self,
        world_size: int,
        reduce_op: callable = cross_device_ops.reduce_add,
        gather_op: callable = cross_device_ops.gather,
    ):
        self.scatter_reduce = ScatterReduce(world_size, reduce_op)
        self.all_gather = AllGather(world_size, gather_op, barrier=False)
        # note: AllGather does not need barrier here because scatter_reduce's ready event serves as barrier

    def __call__(self, x: torch.Tensor, rank: int):
        reduced_part = self.scatter_reduce(x, rank)
        return self.all_gather(reduced_part, rank).view_as(x)


class ScatterReduce(CollectiveOpetationBase):
    def __init__(self, world_size: int, reduce_op: callable = cross_device_ops.reduce_add):
        self.world_size = world_size
        self.tensor_parts = [[] for _ in range(world_size)]
        self.parts_ready = [threading.Event() for _ in range(world_size)]
        self.reduce_op = reduce_op

    def __call__(self, x: torch.Tensor, rank: int):
        try:
            for i, part in enumerate(x.flatten().tensor_split(self.world_size)):
                self.tensor_parts[i].append(part)  # append is thread-safe. thanks, GIL!
                if len(self.tensor_parts[i]) == self.world_size:
                    self.parts_ready[i].set()  # can be called more than once; we don't care

            self.parts_ready[rank].wait()
            reduced_part = self.reduce_op(self.tensor_parts[rank], x.device)
            return reduced_part
        finally:
            # prepare for next forward; each rank clears its own data
            self.tensor_parts[rank].clear()
            self.parts_ready[rank].clear()


class AllGather(CollectiveOpetationBase):
    def __init__(self, world_size: int, gather_op: callable = cross_device_ops.gather, barrier: bool = True):
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size) if barrier else None
        self.parts: List[Optional[torch.Tensor]] = [None for _ in range(world_size)]
        self.ranks_updated = []
        self.parts_ready = threading.Event()
        self.gather_op = gather_op

    def __call__(self, x: torch.Tensor, rank: int):
        if self.barrier is not None:
            self.barrier.wait()  # if this code is ran multiple times in quick succession,
        # this even will wait for the previous call to finish before starting a new one
        parts, ranks_updated, parts_ready = self.parts, self.ranks_updated, self.parts_ready
        # ^-- note: we copy properties to locals so that the "finally" clause is thread-safe
        try:
            parts[rank] = x  # no race b/c each rank writes to a separate location
            ranks_updated.append(rank)  # append is thread-safe. thanks, GIL!
            if len(ranks_updated) == self.world_size:
                parts_ready.set()  # can be called more than once; we dont care
            parts_ready.wait()
            # note: for one of the parts with r == rank, part.to(device) is a no-op
            return self.gather_op(parts, x.device)
        finally:
            if ranks_updated[-1] == rank:
                self.parts = [None for _ in range(self.world_size)]
                self.ranks_updated = []
                self.parts_ready = threading.Event()
            # note: we can safely update these properties because all ranks have
            # copied self.parts_* to locals before passing parts_ready.wait


class DistributedAllReduce(CollectiveOpetationBase):
    def __call__(self, x: torch.Tensor, rank: int):
        all_reduce(x)


class DistributedAllGather(CollectiveOpetationBase):
    def __init__(self, world_size: int, dim: int):
        self.dim = dim
        self.world_size = world_size

    def __call__(self, x: torch.Tensor, rank: int):
        gathered_tensors = [torch.empty_like(x) for _ in range(self.world_size)]
        all_gather(gathered_tensors, x)
        return torch.cat(gathered_tensors, dim=self.dim)
