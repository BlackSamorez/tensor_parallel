"""
A prototype TensorParallel wrapper that works without torch.distributed.
Splits linear, conv and some other layers between GPUs
"""

from tensor_parallel.config import Config
from tensor_parallel.dispatch import convert_state_dict, infer_sharded_device_map, save_tensor_parallel
from tensor_parallel.factory import tensor_parallel
from tensor_parallel.pretrained_model import TensorParallelPreTrainedModel
from tensor_parallel.sharding import Sharded
from tensor_parallel.state_actions import StateAction
from tensor_parallel.tensor_parallel import TensorParallel
