import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import BertModel

from tensor_parallel import TensorParallelPreTrainedModel


@pytest.mark.parametrize("use_config", [False, True])
@pytest.mark.parametrize("model_name", ["bert-base-uncased"])
@pytest.mark.parametrize("devices", [("cpu",) * 2])
def test_lora(use_config, model_name, devices):
    torch.manual_seed(0)

    model = BertModel.from_pretrained(model_name).to(devices[0])
    lora_config = LoraConfig(base_model_name_or_path=model_name, lora_alpha=32, lora_dropout=0.05)
    model = get_peft_model(model, lora_config)

    inp1 = torch.randint(1, 1000, size=(2, 3), device=devices[0])
    inp2 = torch.randint(1, 1000, size=(2, 1), device=devices[0])
    inp3 = torch.randint(1, 1000, size=(2, 2), device=devices[0])

    out1_ref = model(inp1, output_hidden_states=True)
    out2_ref = model(inp2, output_hidden_states=True)
    out3_ref = model(inp3, output_hidden_states=True)

    if not use_config:
        model.config.model_type = "idk"  # pretend it's unknown model

    model_tp = TensorParallelPreTrainedModel(model, devices)
    del model

    out1 = model_tp(inp1, output_hidden_states=True)
    out2 = model_tp(inp2, output_hidden_states=True)
    out3 = model_tp(inp3, output_hidden_states=True)

    torch.testing.assert_close(out1_ref.hidden_states[-1], out1.hidden_states[-1], atol=3e-3, rtol=1e-05)
    torch.testing.assert_close(out2_ref.hidden_states[-1], out2.hidden_states[-1], atol=3e-3, rtol=1e-05)
    torch.testing.assert_close(out3_ref.hidden_states[-1], out3.hidden_states[-1], atol=3e-3, rtol=1e-05)
