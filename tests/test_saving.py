import pytest
import torch
from accelerate import init_empty_weights, load_checkpoint_in_model
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoModel

from tensor_parallel import (
    Config,
    TensorParallel,
    TensorParallelPreTrainedModel,
    convert_state_dict,
    infer_sharded_device_map,
    save_tensor_parallel,
)

PATH_TO_SAVE = "/tmp/"


@pytest.mark.parametrize("devices", [("cpu",) * 2, ("cpu",) * 3])
@pytest.mark.parametrize("model_name", ["bert-base-uncased"])
def test_no_parallelism_zero_3(devices, model_name):
    model = AutoModel.from_pretrained(model_name).to(devices[0])
    model_state_dict = model.state_dict()
    model_tp = TensorParallel(
        model, devices, tensor_parallel_config=Config({}, {}, {}, {}), sharded=True
    )  # zero-3 sharding only
    del model
    with save_tensor_parallel(model_tp):
        model_tp_state_dict = model_tp.state_dict()
    del model_tp

    assert sorted(list(model_state_dict.keys())) == sorted(list(model_tp_state_dict.keys()))

    for name in model_state_dict.keys():
        data = model_state_dict[name]
        data_tp = model_tp_state_dict[name]

        assert data.shape == data_tp.shape

        torch.testing.assert_close(data, data_tp)


@pytest.mark.parametrize("devices", [("cpu",) * 2, ("cpu",) * 3])
@pytest.mark.parametrize("model_name", ["bert-base-uncased", "gpt2", "hf-internal-testing/tiny-random-t5"])
def test_parallelism_no_zero_3(devices, model_name):
    model = AutoModel.from_pretrained(model_name).to(devices[0])
    model_state_dict = model.state_dict()
    model_tp = TensorParallelPreTrainedModel(model, devices, sharded=False)
    del model
    with save_tensor_parallel(model_tp):
        model_tp_state_dict = model_tp.state_dict()
    del model_tp

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
    model_tp = TensorParallelPreTrainedModel(model, devices, sharded=True)
    del model
    with save_tensor_parallel(model_tp):
        model_tp_state_dict = model_tp.state_dict()
    del model_tp

    assert sorted(list(model_state_dict.keys())) == sorted(list(model_tp_state_dict.keys()))

    for name in model_state_dict.keys():
        data = model_state_dict[name]
        data_tp = model_tp_state_dict[name]

        assert data.shape == data_tp.shape, name

        torch.testing.assert_close(data, data_tp, msg=lambda msg: f"{name}:\n{msg}")


@pytest.mark.parametrize("devices", [("cpu",) * 2, ("cpu",) * 3])
@pytest.mark.parametrize(
    "model_name",
    ["bert-base-uncased", "hf-internal-testing/tiny-random-t5", "hf-internal-testing/tiny-random-BloomModel"],
)
@pytest.mark.parametrize("shard_as_pretrained", [True, False])
def test_save_keep_shards(devices, model_name, shard_as_pretrained):
    model = AutoModel.from_pretrained(model_name).to(devices[0])
    if shard_as_pretrained:
        model_tp = TensorParallelPreTrainedModel(model, devices)
    else:
        model_tp = TensorParallel(model, devices)

    model_tp.load_state_dict(model_tp.state_dict())


def get_tensor_parallel(model: torch.nn.Module, devices, pretrained: bool, sharded: bool):
    if pretrained:
        return TensorParallelPreTrainedModel(model, devices, sharded=sharded)
    else:
        return TensorParallel(model, devices, sharded=sharded)


@pytest.mark.parametrize("devices", [("cpu",) * 2, ("cpu",) * 3])
@pytest.mark.parametrize("model_name", ["bert-base-uncased", "hf-internal-testing/tiny-random-BloomModel"])
@pytest.mark.parametrize("pretrained", [True, False])
@pytest.mark.parametrize("sharded", [True, False])
def test_save_shards_load_shards(devices, model_name, pretrained, sharded):
    devices = [torch.device(device) for device in devices]

    model = AutoModel.from_pretrained(model_name).to(devices[0])
    model_tp = get_tensor_parallel(model, devices, pretrained, sharded=sharded)

    if pretrained:
        half_the_model = f"{sum([p.numel() for p in model_tp.parameters()]) * 8 // 1_000_000 // 2}MB"
        model_tp.save_pretrained(PATH_TO_SAVE, max_shard_size=half_the_model)
    else:
        torch.save(model_tp.state_dict(), PATH_TO_SAVE + "test_save_shards_load_shards.bin")
    del model_tp

    new_model_tp = get_tensor_parallel(AutoModel.from_pretrained(model_name), devices, pretrained, sharded=sharded)

    checkpoint = PATH_TO_SAVE + ("pytorch_model.bin.index.json" if pretrained else "test_save_shards_load_shards.bin")
    load_checkpoint_in_model(
        new_model_tp,
        checkpoint=checkpoint,
        device_map=infer_sharded_device_map(new_model_tp),
    )
    assert not "meta" in [p.device.type for p in new_model_tp.parameters()]
    new_model_tp(torch.zeros(1, 8, dtype=int))


@pytest.mark.parametrize("use_pretrained", [False, True])
@pytest.mark.parametrize("devices", [("cpu",) * 2, ("cpu",) * 3])
@pytest.mark.parametrize("model_name", ["bert-base-uncased"])
@pytest.mark.parametrize("sharded", [True, False])
def test_convert_state_dict(use_pretrained, devices, model_name, sharded):
    model = AutoModel.from_pretrained(model_name)
    torch.save(model.state_dict(), PATH_TO_SAVE + "test_convert_state_dict.bin")
    del model

    with init_empty_weights():
        meta_model = AutoModel.from_pretrained(model_name)
        if use_pretrained:
            model_tp = TensorParallelPreTrainedModel(meta_model, devices, sharded=False)
        else:
            model_tp = TensorParallel(meta_model, devices, sharded=False)

    converted_state_dict = convert_state_dict(
        torch.load(PATH_TO_SAVE + "test_convert_state_dict.bin"),
        model_tp.tensor_parallel_config,
        world_size=len(devices),
        for_pretrained=use_pretrained,
    )

    for param_name, param in converted_state_dict.items():
        set_module_tensor_to_device(model_tp, param_name, "cpu", value=param)

    # When using meta device ZeRO-3 should only be applied after dispatch
    if sharded:
        model_tp.apply_sharding()

    model_tp(torch.zeros(1, 8, dtype=int))
