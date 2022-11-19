import torch, torch.nn as nn
import torch.distributed as dist
import re


class SlicingConfig():
    def __init__(self, tensor_rules: dict, module_rules: dict):
        self.tensor_rules = tensor_rules
        self.module_rules = module_rules
                

def slice_weight_vertical(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    assert(tensor.shape[-2] % world_size == 0)
    slice_size = tensor.shape[-2] // world_size

    return tensor[..., rank * slice_size: (rank + 1) * slice_size, :]


def slice_bias_vertical(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    assert(tensor.shape[-1] % world_size == 0)
    slice_size = tensor.shape[-1] // world_size

    return tensor[rank * slice_size: (rank + 1) * slice_size]


def slice_weight_horizontal(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    assert(tensor.shape[-1] % world_size == 0)
    slice_size = tensor.shape[-1] // world_size

    return tensor[..., rank * slice_size: (rank + 1) * slice_size]


def slice_bias_horizontal(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    return tensor / world_size


def slice_tensors(key_parameter_iterator, tensor_rules: dict, rank: int, world_size: int):
    regular_rules = [(re.compile(".*" + key + "\\.(weight|bias)"), value) for key, value in tensor_rules.items()]

    with torch.no_grad():
        for name, param in key_parameter_iterator:
            for pattern, rule in regular_rules:
                if not pattern.search(name) is None:
                    name_ending = name.split('.')[-1]
                    match (rule, name_ending):
                        case ("vertical", "weight"):
                            param.data = slice_weight_vertical(param.data, rank=rank, world_size=world_size)
                        case ("vertical", "bias"):
                            param.data = slice_bias_vertical(param.data, rank=rank, world_size=world_size)
                        case ("horizontal", "weight"):
                            param.data = slice_weight_horizontal(param.data, rank=rank, world_size=world_size)
                        case ("horizontal", "bias"):
                            param.data = slice_bias_horizontal(param.data, rank=rank, world_size=world_size)
                        case _:
                            raise Exception("Fuck you tensor!")


def process_input(rules, rank, world_size, *args, **kwargs):
    args = list(args)
    for target, action in rules.items():
        match action:
            case "cut":
                match target:
                    case int(idx):
                        slice_size= args[idx].shape[0] // world_size
                        args[idx] = args[idx][..., rank * slice_size: (rank + 1) * slice_size, :]
                    case str(name):
                        slice_size= kwargs[name].shape[0] // world_size
                        kwargs[name] = kwargs[name][rank * slice_size: (rank + 1) * slice_size, ...]
                    case _:
                        raise Exception("Fuck you cut input!")

            case "scale":
                match target:
                    case int(idx):
                        args[idx] = args[idx] / world_size
                    case str(name):
                        kwargs[name] = kwargs[name]/ world_size
                    case _:
                        raise Exception("Fuck you scale input!")
            case _:
                raise Exception("Fuck you input action!")

    return args, kwargs


def process_output(output, rules):
    for target, action in rules.items():
        match action:
            case "reduce":
                match target:
                    case "ALL":
                        dist.all_reduce(output)
                    case int(idx):
                        dist.all_reduce(output[idx])
                    case _:
                        raise Exception("Fuck you output taget!")
            case _:
                raise Exception("Fuck you output action!")
    return output


def process_attr(module, rules, rank, world_size):
    for attr, action in rules.items():
            match action:
                case "scale_int":
                    setattr(module, attr, getattr(module, attr) // world_size)


class ParallelLayerWrapper(nn.Module):
    def __init__(self, module: nn.Module, module_rules: dict, rank: int, world_size: int):
        super().__init__()
        self.module = module
        process_attr(self.module, module_rules["attributes"], rank=rank, world_size=world_size)

        self.input_rules = module_rules["input"]
        self.output_rules = module_rules["output"]

        self.rank = rank
        self.world_size = world_size

        
    def forward(self, *args, **kwargs):
        args, kwargs = process_input(self.input_rules, self.rank, self.world_size, *args, **kwargs)
        output = self.module(*args, **kwargs)
        return process_output(output, self.output_rules)


def wrap_submodules(model: nn.Module, module_rules: dict, rank: int, world_size: int):
    unique_wrappers = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            for pattern, rule in module_rules.items():
                if re.search(".*" + pattern + "$", name) is not None:
                    unique_wrappers[module] = ParallelLayerWrapper(module, rule, rank=rank, world_size=world_size)

    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if child in unique_wrappers:
                setattr(parent, child_name, unique_wrappers[child])


def tensor_parallel(model_cls, slicing_config: SlicingConfig, rank: int, world_size: int):
    global SLICING_CONFIG
    SLICING_CONFIG = slicing_config

    global RANK
    RANK = rank

    global WORLD_SIZE
    WORLD_SIZE = world_size

    class _TensorParallel(model_cls):
        def __new__(cls, *args, **kwargs):
            model = model_cls(*args, **kwargs)  # Create an instance of vanilla model
            
            # modify untrained parameters/buffers
            slice_tensors(model.named_parameters(), SLICING_CONFIG.tensor_rules, RANK, WORLD_SIZE)

            return model

        @classmethod
        def _load_pretrained_model(cls, model: model_cls, state_dict, loaded_keys, *args, **kwargs):
            slice_tensors(state_dict.items(), SLICING_CONFIG.tensor_rules, RANK, WORLD_SIZE)
            result = super()._load_pretrained_model(model, state_dict, loaded_keys, *args, **kwargs)
            
            wrap_submodules(model, SLICING_CONFIG.module_rules, RANK, WORLD_SIZE)

            return result
        
    return _TensorParallel
