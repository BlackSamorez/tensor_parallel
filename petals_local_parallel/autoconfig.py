import torch
from torch import nn

from slicing_config import SlicingConfig

def build_default_slicing_config(model: nn.Module) -> SlicingConfig:
    slicing_config = SlicingConfig({}, {})

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            slicing_config.tensor_rules[name + ".(weight|bias)"] = "vertical"
            slicing_config.module_rules[name] = {"input": {}, "output": {0: "gather"}, "attributes": {}}
    
    return slicing_config
