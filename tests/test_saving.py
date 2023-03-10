import pytest
import torch
from accelerate import init_empty_weights
from transformers import BertModel

from tensor_parallel import (
    Config,
    Sharded,
    TensorParallel,
    TensorParallelPreTrainedModel,
    load_separate_shards,
    save_separate_shards,
    tensor_parallel,
)


@pytest.mark.parametrize("devices", [("cpu",) * 2, ("cpu",) * 3])
@pytest.mark.parametrize("model_name", ["bert-base-uncased"])
def test_no_parallelism_zero_3(devices, model_name):
    model = BertModel.from_pretrained(model_name).to(devices[0])
    model_state_dict = model.state_dict()
    model_tp = Sharded(TensorParallel(model, devices, config=Config({}, {}, {}, {})))  # zero-3 sharding only
    model_tp_state_dict = model_tp.state_dict()

    assert sorted(list(model_state_dict.keys())) == sorted(list(model_tp_state_dict.keys()))

    for name in model_state_dict.keys():
        data = model_state_dict[name]
        data_tp = model_tp_state_dict[name]

        assert data.shape == data_tp.shape

        torch.testing.assert_close(data, data_tp)


@pytest.mark.parametrize("devices", [("cpu",) * 2, ("cpu",) * 3])
@pytest.mark.parametrize("model_name", ["bert-base-uncased"])
def test_parallelism_no_zero_3(devices, model_name):
    model = BertModel.from_pretrained(model_name).to(devices[0])
    model_state_dict = model.state_dict()
    model_tp = TensorParallelPreTrainedModel(model, devices)
    model_tp_state_dict = model_tp.state_dict()

    assert sorted(list(model_state_dict.keys())) == sorted(list(model_tp_state_dict.keys()))

    for name in model_state_dict.keys():
        data = model_state_dict[name]
        data_tp = model_tp_state_dict[name]

        assert data.shape == data_tp.shape

        torch.testing.assert_close(data, data_tp)


@pytest.mark.parametrize("devices", [("cpu",) * 2, ("cpu",) * 3])
@pytest.mark.parametrize("model_name", ["bert-base-uncased"])
def test_parallelism_zero_3(devices, model_name):
    model = BertModel.from_pretrained(model_name).to(devices[0])
    model_state_dict = model.state_dict()
    model_tp = tensor_parallel(model, devices, sharded=True)
    model_tp_state_dict = model_tp.state_dict()

    assert sorted(list(model_state_dict.keys())) == sorted(list(model_tp_state_dict.keys()))

    for name in model_state_dict.keys():
        data = model_state_dict[name]
        data_tp = model_tp_state_dict[name]

        assert data.shape == data_tp.shape

        torch.testing.assert_close(data, data_tp, msg=lambda msg: f"{name}:\n{msg}")


@pytest.mark.parametrize("devices", [("cpu",) * 2, ("cpu",) * 3])
@pytest.mark.parametrize("model_name", ["bert-base-uncased"])
@pytest.mark.parametrize("shard_as_pretrained", [True, False])
def test_save_keep_shards(devices, model_name, shard_as_pretrained):
    model = BertModel.from_pretrained(model_name).to(devices[0])
    if shard_as_pretrained:
        model_tp = TensorParallelPreTrainedModel(model, devices)
    else:
        model_tp = TensorParallel(model, devices)

    model_tp.set_preserve_shards_when_saving(True)
    model_tp.load_state_dict(model_tp.state_dict())


@pytest.mark.parametrize("devices", [("cpu",) * 2, ("cpu",) * 3])
@pytest.mark.parametrize("model_name", ["bert-base-uncased"])
@pytest.mark.parametrize("shard_as_pretrained", [True, False])
def test_save_shards_load_shards(devices, model_name, shard_as_pretrained):
    devices = [torch.device(device) for device in devices]

    model = BertModel.from_pretrained(model_name).to(devices[0])
    if shard_as_pretrained:
        model_tp = TensorParallelPreTrainedModel(model, devices)
    else:
        model_tp = TensorParallel(model, devices)

    save_separate_shards(model_tp, "/tmp")
    del model_tp

    with init_empty_weights():
        model_tp = TensorParallelPreTrainedModel(BertModel.from_pretrained(model_name), ("meta",) * len(devices))

    load_separate_shards(model_tp, "/tmp/index.json", devices=devices)
