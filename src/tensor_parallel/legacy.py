import logging
from typing import Collection, Optional, Tuple

from torch import nn

from tensor_parallel.pretrained_model import TensorParallelPreTrainedModel
from tensor_parallel.tensor_parallel import TensorParallel

logger = logging.getLogger(__file__)


class Sharded(nn.Module):
    def __new__(
        cls,
        module: Tuple[TensorParallel, TensorParallelPreTrainedModel],
        sharded_param_names: Optional[Collection[str]] = None,
    ):
        logger.warning(f"`Sharded` is deprecated. Please use `.apply_sharding()` method")
        module.apply_sharding(sharded_param_names)
        return module
