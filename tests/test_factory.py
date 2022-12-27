import pytest
import torch.nn as nn
import transformers

from tensor_parallel import TensorParallel, tensor_parallel


@pytest.mark.parametrize("sharded", [True, False])
def test_factory_nn_module(sharded):
    model = nn.Sequential(
        nn.Embedding(num_embeddings=1337, embedding_dim=64),
        nn.LayerNorm(64),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    assert isinstance(model, nn.Module)
    model = tensor_parallel(model, device_ids=["cpu", "cpu"], sharded=sharded)
    assert isinstance(model, TensorParallel) if not sharded else isinstance(model.module, TensorParallel)


def test_factory_pretrainedmodel():
    devices = ["cpu", "cpu"]
    model_name = "bigscience/bloom-560m"
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).float().to(devices[0])

    assert isinstance(model, transformers.PreTrainedModel)
    model = tensor_parallel(model, device_ids=["cpu", "cpu"])
    assert isinstance(model, transformers.PreTrainedModel)
