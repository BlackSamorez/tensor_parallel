import torch
from transformers import BertModel, PreTrainedModel

from tensor_parallel import Sharded, tensor_parallel


def test_legacy_factory_and_sharded():
    model = BertModel.from_pretrained("bert-base-uncased")

    tp_model = tensor_parallel(model, sharded=False)
    assert isinstance(tp_model, PreTrainedModel)
    tp_model.wrapped_model = Sharded(tp_model.wrapped_model)

    tp_model(torch.zeros(1, 8, dtype=int))
