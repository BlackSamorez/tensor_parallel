from typing import Sequence

import torch
import torch.distributed
from torch import nn
from torch.nn.modules import conv

from tensor_parallel.config import Config
from tensor_parallel.state_actions import Scale, Split, SplitInGroupedChunks


def get_default_config(module: nn.Module, device_ids: Sequence[torch.device]) -> Config:
    """Make a generic config that wraps individual linear, embedding and convolutional layers"""
    emb_weights = {m.weight for m in module.modules() if isinstance(m, (nn.Embedding, nn.EmbeddingBag))}

    state_rules = {}
    input_rules = {}
    output_rules = {}
    for name, module in module.named_modules():
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
            assert module.max_norm is None or module.norm_type < 2
            assert getattr(module, "bias", None) is None or module.bias.shape == module.embedding_dim
            state_rules[f"^{name}.weight$"] = Split(world_size=len(device_ids), dim=1)
            if hasattr(module, "bias"):
                state_rules[f"^{name}.bias$"] = Split(world_size=len(device_ids), dim=0)
            output_rules[f"^{name}$"] = {0: "gather -1"}
        elif isinstance(module, nn.Linear) and "lora_A" not in name and "lora_B" not in name:
            assert module.weight.shape == (module.out_features, module.in_features)
            assert module.bias is None or module.bias.shape == (module.out_features,)
            if module.weight not in emb_weights:  # regular linear layer
                state_rules[f"^{name}.(weight|bias)$"] = Split(world_size=len(device_ids), dim=0)
                output_rules[f"^{name}$"] = {0: "gather -1"}
            else:
                # linear weight tied with embeddings; this is a popular special case for language models;
                # since embedding weight will be sliced over dim 1, we should adapt to the input-sliced weight
                input_rules[f"^{name}$"] = {0: Split(world_size=len(device_ids), dim=-1)}
                output_rules[f"^{name}$"] = {0: "sum"}
                if module.bias is not None:
                    state_rules[f"^{name}.bias$"] = Scale(world_size=len(device_ids))
        elif isinstance(module, conv._ConvNd) and module.groups == 1:
            shape = [module.out_channels, module.in_channels] + list(module.kernel_size)
            shape[:2] = shape[:2][::-1] if module.transposed else shape[:2]
            shape = tuple(shape)
            assert module.weight.shape == shape, f"{module.weight.shape} != {shape}"
            assert module.bias is None or module.bias.shape == (module.out_channels,), module.bias.shape
            state_rules[f"^{name}.weight$"] = (
                Split(world_size=len(device_ids), dim=1)
                if module.transposed
                else Split(world_size=len(device_ids), dim=0)
            )
            if module.bias is not None:
                state_rules[f"^{name}.bias$"] = Split(world_size=len(device_ids), dim=0)
            output_rules[f"^{name}$"] = {0: "gather 1"}
        elif isinstance(module, conv._ConvNd) and module.groups != 1:
            # group conv: split each group individually over input channels to avoid changing module.groups
            groups = module.groups
            shape = [module.out_channels // groups, module.in_channels // groups] + list(module.kernel_size)
            shape[:2] = shape[:2][::-1] if module.transposed else shape[:2]
            shape[0] *= module.groups
            shape = tuple(shape)
            assert module.weight.shape == shape, f"{module.weight.shape} != {shape}"
            assert module.bias is None or module.bias.shape == (module.out_channels,), module.bias.shape
            if not module.transposed:
                state_rules[f"^{name}.weight$"] = Split(world_size=len(device_ids), dim=1)
            else:
                state_rules[f"^{name}.weight$"] = SplitInGroupedChunks(
                    world_size=len(device_ids), dim=0, num_groups=groups, chunk_size=1
                )
            if module.bias is not None:
                state_rules[f"^{name}.bias$"] = Scale(world_size=len(device_ids))
            input_rules[f"^{name}$"] = {
                0: SplitInGroupedChunks(world_size=len(device_ids), dim=1, num_groups=groups, chunk_size=1)
            }
            output_rules[f"^{name}$"] = {0: "sum"}
    return Config(state_rules, input_rules, output_rules, {})
