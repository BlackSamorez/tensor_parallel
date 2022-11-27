from slicer_wrapper import SlicingConfig, slice_tensors, wrap_submodules
from slicing_configs import SLICING_CONFIGS
import torch.distributed as dist

def tensor_parallel(model_cls, slicing_config: SlicingConfig = None, rank: int = None, world_size: int = None):
    if slicing_config in None:
        for model_class, config in SLICING_CONFIGS.values():
            if model_cls.__name__ == model_class:
                slicing_config = config
                break
        else:
            raise NotImplemented("Unknown model and lazy mode not implemented yet. Must specify config")

    if rank in None:
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            raise Exception("Rank must be specified if torch.distributed is not initialized")

    if world_size in None:
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            raise Exception("World size must be specified if torch.distributed is not initialized")
    class _TensorParallel(model_cls):
        slicing_config = None
        rank = None
        world_size = None
        def __new__(cls, *args, __slicing_config=slicing_config, __rank=rank, __world_size=world_size, **kwargs):
            _TensorParallel.slicing_config = __slicing_config
            _TensorParallel.rank = __rank
            _TensorParallel.world_size = __world_size

            model = model_cls(*args, **kwargs)  # Create an instance of vanilla model
            
            # modify untrained parameters/buffers
            slice_tensors(model.named_parameters(), slicing_config.tensor_rules, rank, world_size)
            return model

        @classmethod
        def _load_pretrained_model(cls, model: model_cls, state_dict, loaded_keys, *args, **kwargs):
            slice_tensors(state_dict.items(), slicing_config.tensor_rules, rank, world_size)
            result = super()._load_pretrained_model(model, state_dict, loaded_keys, *args, **kwargs)
            
            wrap_submodules(model, slicing_config.module_rules, rank, world_size)

            return result
        
    return _TensorParallel
