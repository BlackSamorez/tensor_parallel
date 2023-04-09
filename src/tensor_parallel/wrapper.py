"""
Tensor parallelism config and functions for splitting model into shards
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.distributed
from torch import nn

from tensor_parallel.config import MappedActions
from tensor_parallel.state_actions import StateAction


def apply_action(input: torch.Tensor, action: StateAction, *, rank: int, world_size: int):
    if isinstance(action, tuple):
        action = action[0]  # get splitting action
    if callable(action):
        return action(input, rank=rank)  # allreduce/allgather or a custom user-defined callable
    action_type, *opts = action.split()
    if action_type == "split":
        dim = int(opts[0])
        return torch.tensor_split(input, world_size, dim=dim)[rank]
    if action_type == "scale":
        return input / world_size
    raise Exception(f"unexpected action {action_type}; supported actions: split, scale, or custom user-defined")


def process_input(input_actions: MappedActions, rank: int, world_size: int, *args, **kwargs):
    extended_kwargs = dict(kwargs)
    extended_kwargs.update(enumerate(args))
    for target, action in input_actions.items():
        extended_kwargs[target] = apply_action(extended_kwargs.get(target), action, rank=rank, world_size=world_size)
    args = [extended_kwargs.pop(i) for i in range(len(args))]
    return args, extended_kwargs


def process_output(output, output_actions: MappedActions, *, rank: int, world_size: int):
    if isinstance(output, torch.Tensor):
        return process_output({0: output}, output_actions, rank=rank, world_size=world_size)[0]
    if isinstance(output, Sequence):
        output_dict = process_output(dict(enumerate(output)), output_actions, rank=rank, world_size=world_size)
        return type(output)((output_dict[i] for i in range(len(output))))
    for target, action in output_actions.items():
        output[target] = apply_action(output.get(target), action, rank=rank, world_size=world_size)
    return output


class TensorParallelWrapper(nn.Module):
    """Wraps a single module, applies tensor parallelism actions to module inputs and outputs"""

    def __init__(
        self,
        module: nn.Module,
        input_actions: MappedActions,
        output_actions: MappedActions,
        *,
        rank: int,
        world_size: int,
    ):
        super().__init__()
        self.tp_wrapped_module = module
        self.__dict__["tp_wrapped_module"] = module  # for it to be accessible without getattr
        self.input_actions, self.output_actions = input_actions, output_actions
        self.rank, self.world_size = rank, world_size

    def forward(self, *args, **kwargs):
        args, kwargs = process_input(self.input_actions, self.rank, self.world_size, *args, **kwargs)
        output = self.tp_wrapped_module(*args, **kwargs)
        return process_output(output, self.output_actions, rank=self.rank, world_size=self.world_size)

    def __getattr__(self, attr):
        return getattr(self.tp_wrapped_module, attr)
