"""
The TensorParallel module wrapper for Hugging Face PreTrainedModel
"""
import logging
from functools import lru_cache
from typing import Any, Dict, Optional, Sequence

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from tensor_parallel.config import TENSOR_PARALLEL_USE_NATIVE, Config
from tensor_parallel.per_device_tensors import PerDeviceTensors
from tensor_parallel.sharding import Sharded
from tensor_parallel.slicing_configs import PREDEFINED_CONFIGS
from tensor_parallel.tensor_parallel import TensorParallel, check_device_ids, parallel_apply, parallel_apply_simple
from tensor_parallel.utils import nested_map

logger = logging.getLogger(__file__)


def find_predefined_tensor_parallel_config(
    model_config: PretrainedConfig, device_ids: Optional[Sequence[torch.device]]
) -> Optional[Config]:
    device_ids = check_device_ids(device_ids)

    try:
        return PREDEFINED_CONFIGS[model_config.model_type](model_config, device_ids)
    except KeyError:
        logger.warning(
            "Using automatic config: tensor parallel config not provided and no custom config registered for the model"
        )
        return None


class TensorParallelPreTrainedModel(PreTrainedModel):
    is_parallelizable = model_parallel = True

    def __init__(
        self,
        module: PreTrainedModel,
        device_ids: Optional[Sequence[torch.device]] = None,
        output_device: Optional[torch.device] = None,
        output_device_index: Optional[int] = None,
        tensor_parallel_config: Optional[Config] = None,
        distributed: bool = True,
    ):
        super().__init__(module.config)  # Temporary empty config. Gets replaced in from_pretrained

        if hasattr(module, "_hf_hook"):
            from accelerate.hooks import remove_hook_from_module

            remove_hook_from_module(module, recurse=True)

        if tensor_parallel_config is None:
            tensor_parallel_config = find_predefined_tensor_parallel_config(module.config, device_ids)

        self.wrapped_model = TensorParallel(
            module, device_ids, output_device, output_device_index, tensor_parallel_config, distributed=distributed
        )

    @property
    def devices(self):
        return self.wrapped_model.devices

    @property
    def tensor_parallel_config(self):
        return self.wrapped_model.tensor_parallel_config

    @property
    def preserve_shards_when_saving(self):
        return self.wrapped_model.preserve_shards_when_saving

    @preserve_shards_when_saving.setter
    def preserve_shards_when_saving(self, value):
        self.wrapped_model.preserve_shards_when_saving = value

    def forward(self, *args, **kwargs):
        return self.wrapped_model(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        if self.wrapped_model.preserve_shards_when_saving:
            return state_dict

        prefix = kwargs["prefix"] if "prefix" in kwargs else ""
        module_prefix = prefix + "wrapped_model."

        module_parameter_names = [name for name in state_dict.keys() if name.startswith(module_prefix)]
        for module_parameter_name in module_parameter_names:
            original_parameter_name = prefix + module_parameter_name[len(module_prefix) :]
            state_dict[original_parameter_name] = state_dict[module_parameter_name]
            del state_dict[module_parameter_name]

        return state_dict

    def _validate_model_class(self):
        return self.wrapped_model.module_shards[0]._validate_model_class()

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        return self.wrapped_model.module_shards[0]._validate_model_kwargs(model_kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.wrapped_model.module_shards[0].prepare_inputs_for_generation(*args, **kwargs)

    def _reorder_cache(self, past, beam_idx):
        for i, shard in enumerate(self.wrapped_model.module_shards):
            shard._reorder_cache(
                nested_map(lambda x: x[i] if isinstance(x, PerDeviceTensors) else x, past),
                beam_idx.to(self.wrapped_model.devices[i]),
            )

    @lru_cache(maxsize=None)
    def get_encoder(self):
        assert len(self.wrapped_model.module_shards), "Can't get encoder since no module shards present"
        if len(self.wrapped_model.module_shards) == 1:
            return self.wrapped_model.module_shards[0].get_encoder()

        encoder_shards = [
            encoder_decoder_shard.get_encoder() for encoder_decoder_shard in self.wrapped_model.module_shards
        ]

        encoder_wrapper_class = None
        if isinstance(self.wrapped_model, TensorParallel):

            class _EncoderWrapper(torch.nn.Module):
                def __init__(self, wrapped_pretrained_model: TensorParallelPreTrainedModel) -> None:
                    super().__init__()
                    self.wrapped_pretrained_model = wrapped_pretrained_model

                def forward(self, *args, **kwargs):
                    (
                        inputs,
                        kwargs_tup,
                    ) = self.wrapped_pretrained_model.wrapped_model.prepare_args_kwargs_for_forward(*args, **kwargs)
                    if self.wrapped_pretrained_model.wrapped_model.all_cuda and not TENSOR_PARALLEL_USE_NATIVE:
                        return parallel_apply(
                            encoder_shards,
                            inputs,
                            kwargs_tup,
                            self.wrapped_pretrained_model.wrapped_model.devices,
                        )[self.wrapped_pretrained_model.wrapped_model.output_device_index]
                    else:
                        return parallel_apply_simple(
                            encoder_shards,
                            inputs,
                            kwargs_tup,
                            self.wrapped_pretrained_model.wrapped_model.devices,
                        )[self.wrapped_pretrained_model.wrapped_model.output_device_index]

            encoder_wrapper_class = _EncoderWrapper

        elif isinstance(self.wrapped_model, Sharded):

            class _EncoderWrapper(torch.nn.Module):
                def __init__(self, wrapped_pretrained_model: TensorParallelPreTrainedModel) -> None:
                    super().__init__()
                    self.wrapped_pretrained_model = wrapped_pretrained_model

                def forward(self, *args, **kwargs):
                    if (
                        len(self.wrapped_pretrained_model.wrapped_model.module.module_shards) > 1
                        and len(self.wrapped_pretrained_model.wrapped_model.sharded_param_names) > 0
                    ):
                        self.wrapped_pretrained_model.wrapped_model._maybe_fill_sharded_params()
                    (
                        inputs,
                        kwargs_tup,
                    ) = self.wrapped_pretrained_model.wrapped_model.module.prepare_args_kwargs_for_forward(
                        *args, **kwargs
                    )
                    if self.wrapped_pretrained_model.wrapped_model.module.all_cuda and not TENSOR_PARALLEL_USE_NATIVE:
                        return parallel_apply(
                            encoder_shards,
                            inputs,
                            kwargs_tup,
                            self.wrapped_pretrained_model.wrapped_model.module.devices,
                        )[self.wrapped_pretrained_model.wrapped_model.module.output_device_index]
                    else:
                        return parallel_apply_simple(
                            encoder_shards,
                            inputs,
                            kwargs_tup,
                            self.wrapped_pretrained_model.wrapped_model.module.devices,
                        )[self.wrapped_pretrained_model.wrapped_model.module.output_device_index]

            encoder_wrapper_class = _EncoderWrapper

        return encoder_wrapper_class(self)
