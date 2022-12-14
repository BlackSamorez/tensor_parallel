"""
The TensorParallel module wrapper for Hugging Face PreTrainedModel
"""
import logging
from typing import Any, Dict, Optional, Sequence

import torch
import transformers
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from tensor_parallel.sharding import Sharded
from tensor_parallel.slicer_wrapper import TENSOR_PARALLEL_USE_NATIVE, Config
from tensor_parallel.slicing_configs import PREDEFINED_CONFIGS
from tensor_parallel.tensor_parallel import (
    PerDeviceTensors,
    TensorParallel,
    check_device_ids,
    parallel_apply,
    parallel_apply_simple,
)
from tensor_parallel.utils import nested_map

logger = logging.getLogger(__file__)


def find_predefined_tensor_parallel_config(
    model_config: PretrainedConfig, device_ids: Optional[Sequence[torch.device]]
) -> Optional[Config]:
    device_ids = check_device_ids(device_ids)
    if len(model_config.architectures) != 1:
        logger.warning(
            f"Using automatic config: no tensor parallel config provided and model architectures list is ambigious: {model_config.architectures}"
        )
        return None

    try:
        return PREDEFINED_CONFIGS[model_config.architectures[0]](model_config, device_ids)
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
        config: Optional[Config] = None,
    ):
        super().__init__(module.config)  # Temporary empty config. Gets replaced in from_pretrained

        if config is None:
            config = find_predefined_tensor_parallel_config(module.config, device_ids)

        self.wrapped_model = TensorParallel(module, device_ids, output_device, output_device_index, config)

        if hasattr(self.wrapped_model.module_shards[0], "_hf_hook"):
            from accelerate.hooks import remove_hook_from_module

            remove_hook_from_module(self.wrapped_model, recurse=True)

        self.encoder_shards = nn.ModuleList()
        if module.config.is_encoder_decoder:
            for encoder_decoder_shard in self.wrapped_model.module_shards:
                self.encoder_shards.append(encoder_decoder_shard.get_encoder())

    def forward(self, *args, **kwargs):
        return self.wrapped_model(*args, **kwargs)

    def _validate_model_class(self):
        return self.wrapped_model.module_shards[0]._validate_model_class()

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        return self.wrapped_model.module_shards[0]._validate_model_kwargs(model_kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.wrapped_model.module_shards[0].prepare_inputs_for_generation(*args, **kwargs)

    def _reorder_cache(self, past, beam_idx):
        for i, shard in enumerate(self.wrapped_model.module_shards):
            shard._reorder_cache(nested_map(lambda x: x[i] if isinstance(x, PerDeviceTensors) else x, past), beam_idx)

    def get_encoder(self):
        assert len(self.wrapped_model.module_shards), "Can't get encoder since no module shards present"
        if len(self.wrapped_model.module_shards) == 1:
            return self.wrapped_model.module_shards[0].get_encoder()

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
                            self.wrapped_pretrained_model.encoder_shards,
                            inputs,
                            kwargs_tup,
                            self.wrapped_pretrained_model.wrapped_model.devices,
                        )[self.wrapped_pretrained_model.wrapped_model.output_device_index]
                    else:
                        return parallel_apply_simple(
                            self.wrapped_pretrained_model.encoder_shards,
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
                            self.wrapped_pretrained_model.encoder_shards,
                            inputs,
                            kwargs_tup,
                            self.wrapped_pretrained_model.wrapped_model.module.devices,
                        )[self.wrapped_pretrained_model.wrapped_model.module.output_device_index]
                    else:
                        return parallel_apply_simple(
                            self.wrapped_pretrained_model.encoder_shards,
                            inputs,
                            kwargs_tup,
                            self.wrapped_pretrained_model.wrapped_model.module.devices,
                        )[self.wrapped_pretrained_model.wrapped_model.module.output_device_index]

            encoder_wrapper_class = _EncoderWrapper

        return encoder_wrapper_class(self)
