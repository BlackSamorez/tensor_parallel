import pytest
import torch
from accelerate import init_empty_weights, load_checkpoint_in_model
from transformers import AutoConfig, AutoModel

from tensor_parallel import (
    Config,
    Sharded,
    TensorParallel,
    TensorParallelPreTrainedModel,
    convert_state_dict,
    infer_sharded_device_map,
    save_tensor_parallel,
    tensor_parallel,
)
from tensor_parallel.pretrained_model import find_predefined_tensor_parallel_config

PATH_TO_SAVE = "/tmp/"


@pytest.mark.parametrize("devices", [("cpu",) * 2, ("cpu",) * 3])
@pytest.mark.parametrize("model_name", ["bert-base-uncased"])
def test_no_parallelism_zero_3(devices, model_name):
    model = AutoModel.from_pretrained(model_name).to(devices[0])
    model_state_dict = model.state_dict()
    model_tp = Sharded(
        TensorParallel(model, devices, tensor_parallel_config=Config({}, {}, {}, {}))
    )  # zero-3 sharding only
    with save_tensor_parallel(model_tp):
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
    model = AutoModel.from_pretrained(model_name).to(devices[0])
    model_state_dict = model.state_dict()
    model_tp = TensorParallelPreTrainedModel(model, devices)
    with save_tensor_parallel(model_tp):
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
    model = AutoModel.from_pretrained(model_name).to(devices[0])
    model_state_dict = model.state_dict()
    model_tp = tensor_parallel(model, devices, sharded=True)
    with save_tensor_parallel(model_tp):
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
    model = AutoModel.from_pretrained(model_name).to(devices[0])
    if shard_as_pretrained:
        model_tp = TensorParallelPreTrainedModel(model, devices)
    else:
        model_tp = TensorParallel(model, devices)

    model_tp.load_state_dict(model_tp.state_dict())


def test_sharding_meta():
    model_name = "bert-base-uncased"
    with init_empty_weights():
        model_tp = TensorParallel(AutoModel.from_pretrained(model_name), ["meta", "meta"])

    with pytest.raises(RuntimeError):
        Sharded(model_tp)


@pytest.mark.parametrize("devices", [("cpu",) * 2, ("cpu",) * 3])
@pytest.mark.parametrize("model_name", ["bert-base-uncased"])
@pytest.mark.parametrize("pretrained", [True])
def test_save_shards_load_shards(devices, model_name, pretrained):
    devices = [torch.device(device) for device in devices]

    model = AutoModel.from_pretrained(model_name).to(devices[0])
    shraded_class = TensorParallelPreTrainedModel if pretrained else TensorParallel
    model_tp = shraded_class(model, devices)

    if pretrained:
        half_the_model = f"{sum([p.numel() for p in model_tp.parameters()]) // 1_000_000 // 2}MB"
        model_tp.save_pretrained(PATH_TO_SAVE, max_shard_size=half_the_model)
    else:
        torch.save(model_tp.state_dict(), PATH_TO_SAVE + "test_save_shards_load_shards.bin")
    del model_tp

    with init_empty_weights():
        model_tp = shraded_class(AutoModel.from_config(AutoConfig.from_pretrained(model_name)), devices)

    checkpoint = PATH_TO_SAVE + ("pytorch_model.bin.index.json" if pretrained else "test_save_shards_load_shards.bin")
    load_checkpoint_in_model(
        model_tp,
        checkpoint=checkpoint,
        device_map=infer_sharded_device_map(model_tp),
    )
    assert not "meta" in [p.device.type for p in model_tp.parameters()]


@pytest.mark.parametrize("use_pretrained", [False, True])
@pytest.mark.parametrize("devices", [("cpu",) * 2, ("cpu",) * 3])
@pytest.mark.parametrize("model_name", ["bert-base-uncased"])
def test_convert_state_dict(use_pretrained, devices, model_name):
    model = AutoModel.from_pretrained(model_name).to(devices[0])
    torch.save(model.state_dict(), PATH_TO_SAVE + "test_convert_state_dict.bin")

    if use_pretrained:
        model_tp = TensorParallelPreTrainedModel(model, devices)
    else:
        model_tp = TensorParallel(model, devices)
    del model

    model_tp_state_dict = model_tp.state_dict()
    converted_state_dict = convert_state_dict(
        torch.load(PATH_TO_SAVE + "test_convert_state_dict.bin"),
        model_tp.tensor_parallel_config,
        world_size=len(devices),
        for_pretrained=use_pretrained,
    )

    assert sorted(list(model_tp_state_dict.keys())) == sorted(list(converted_state_dict.keys()))

    for name in model_tp_state_dict.keys():
        data_tp = model_tp_state_dict[name]
        data_converted = converted_state_dict[name]

        assert data_tp.shape == data_converted.shape

        torch.testing.assert_close(data_tp, data_converted)
