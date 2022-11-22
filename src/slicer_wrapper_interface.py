from slicer_wrapper import SlicingConfig, slice_tensors, wrap_submodules

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