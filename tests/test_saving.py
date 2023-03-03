import pytest
from transformers import BertModel

from tensor_parallel import Config, Sharded, TensorParallel


@pytest.mark.parametrize("devices", [("cpu",) * 2])
@pytest.mark.parametrize("model_name", ["bert-base-uncased"])
def test_zero_3(devices, model_name):
    model = BertModel.from_pretrained(model_name).to(devices[0])
    model_state_dict = model.state_dict()
    model_tp = Sharded(TensorParallel(model, devices, config=Config({}, {}, {}, {})))  # zero-3 sharding only
    model_tp_state_dict = model_tp.state_dict()

    assert sorted(list(model_state_dict.keys())) == sorted(list(model_tp_state_dict.keys()))