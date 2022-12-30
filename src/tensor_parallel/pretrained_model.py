"""
The TensorParallel module wrapper for Hugging Face PreTrainedModel
"""
import logging
from typing import Any, Dict, Optional, Sequence

import torch
import transformers
from transformers import PretrainedConfig, PreTrainedModel

from tensor_parallel.slicer_wrapper import Config
from tensor_parallel.slicing_configs import PREDEFINED_CONFIGS
from tensor_parallel.tensor_parallel import TensorParallel, check_device_ids

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

    def forward(self, *args, **kwargs):
        return self.wrapped_model(*args, **kwargs)

    def _validate_model_class(self):
        return self.wrapped_model.module_shards[0]._validate_model_class()

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        return self.wrapped_model.module_shards[0]._validate_model_kwargs(model_kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.wrapped_model.module_shards[0].prepare_inputs_for_generation(*args, **kwargs)

    def _reorder_cache(self, past, beam_idx):
        for shard in self.wrapped_model.module_shards:
            shard._reorder_cache(past, beam_idx)
