"""
Utility functions that can process tensors that are located on multiple devices, with autograd support
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch.cuda import nccl
from torch.nn.parallel import comm
from torch.nn.parallel._functions import Broadcast, Gather


def broadcast_coalesced(
    tensors: Sequence[torch.Tensor], devices: Sequence[torch.device], *, all_cuda: bool = None, **kwargs
) -> Sequence[Sequence[torch.Tensor]]:
    if all_cuda is None:
        all_cuda = all(device.type == "cuda" for device in devices)
    if not all_cuda:
        broadcasted = [list() for _ in devices]
        for x in tensors:
            for i, device in enumerate(devices):
                broadcasted[i].append(x.to(device, non_blocking=True))
        return broadcasted
    flat_outputs = Broadcast.apply(devices, *tensors)
    return [flat_outputs[i * len(tensors) : (i + 1) * len(tensors)] for i in range(len(devices))]


def gather(
    tensors: Sequence[torch.Tensor], dim: int = 0, destination: Optional[torch.device] = None, all_cuda: bool = None
):
    """Gather tensors from multiple devices; differentiable w.r.t. input tensors"""
    if all_cuda is None:
        all_cuda = all(x.device.type == "cuda" for x in tensors)
    if destination is None:
        destination = tensors[0].device
    if not all_cuda:
        return torch.cat([x.to(destination, non_blocking=True) for x in tensors], dim=dim)
    return Gather.apply(destination, dim, *tensors)


def reduce_add(tensors: Sequence[torch.Tensor], destination: Optional[torch.device] = None, all_cuda: bool = None):
    if all_cuda is None:
        all_cuda = all(x.device.type == "cuda" for x in tensors)
    if destination is None:
        destination = tensors[0].device
    if not all_cuda:
        return sum([tensor.to(destination, non_blocking=True) for tensor in tensors])
    return _ReduceAdd.apply(destination, *tensors)


def all_gather(tensors: Sequence[torch.Tensor], all_cuda: bool = None) -> Sequence[torch.Tensor]:
    """Perform AllGather on tensors, return stacked tensors on each device; differentiable w.r.t. input tensors"""
    if all_cuda is None:
        all_cuda = all(x.device.type == "cuda" for x in tensors)
    if all_cuda and nccl.is_available(tensors):
        return NCCLAllGatherFunction.apply(*tensors)
    else:
        tensors = tuple(x[None, ...] for x in tensors)  # un-squeeze so that gather(concat) performs stacking
        return tuple(gather(tensors, dim=0, destination=output.device, all_cuda=all_cuda) for output in tensors)


class _ReduceAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, destination: torch.device, *tensors: torch.Tensor):
        ctx.source_gpus = [tensor.get_device() for tensor in tensors]
        return comm.reduce_add(tensors, destination)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None,) + Broadcast.apply(ctx.source_gpus, *grad_outputs)


class NCCLAllReduceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs: torch.Tensor):
        inputs = tuple(map(torch.Tensor.contiguous, inputs))
        assert nccl.is_available(inputs)
        outputs = tuple(map(torch.empty_like, inputs))
        nccl.all_reduce(inputs, outputs, op=nccl.SUM)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        grad_outputs = tuple(map(torch.Tensor.contiguous, grad_outputs))
        assert nccl.is_available(grad_outputs)
        grad_inputs = tuple(map(torch.empty_like, grad_outputs))
        nccl.all_reduce(grad_outputs, grad_inputs, op=nccl.SUM)
        return grad_inputs


class NCCLAllGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs: torch.Tensor):
        world_size = len(inputs)
        inputs = tuple(map(torch.Tensor.contiguous, inputs))
        assert nccl.is_available(inputs)
        outputs = tuple(torch.empty((world_size,) + x.shape, device=x.device, dtype=x.dtype) for x in inputs)
        nccl.all_gather(inputs, outputs)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        grad_outputs = tuple(map(torch.Tensor.contiguous, grad_outputs))
        assert nccl.is_available(grad_outputs)
        grad_inputs = tuple(torch.empty(x.shape[1:], device=x.device, dtype=x.dtype) for x in grad_outputs)
        nccl.reduce_scatter(grad_outputs, grad_inputs, op=nccl.SUM)
        return grad_inputs
