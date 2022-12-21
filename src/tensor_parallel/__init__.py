"""
A prototype TensorParallel wrapper that works without torch.distributed.
Splits linear, conv and some other layers between GPUs
"""

from tensor_parallel.factory import tensor_parallel
from tensor_parallel.slicer_wrapper import Config
from tensor_parallel.tensor_parallel import TensorParallel
from tensor_parallel.tensor_parallel_pretrained_model import TensorParallelPreTrainedModel
