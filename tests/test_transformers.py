import pytest
import torch
import transformers
from transformers import AutoTokenizer

from tensor_parallel import TensorParallel, TensorParallelPreTrainedModel
from tensor_parallel.slicing_configs import get_bloom_config


@pytest.mark.parametrize("use_config", [False, True])
@pytest.mark.parametrize("devices", [("cpu",) * 2, ("cpu",) * 3])
def test_bloom_inference(use_config, devices, model_name="bigscience/bloom-560m"):
    model_config = transformers.AutoConfig.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).float().to(devices[0])

    inp1 = torch.randint(1, 1000, size=(2, 3), device=devices[0])
    inp2 = torch.randint(1, 1000, size=(2, 1), device=devices[0])
    inp3 = torch.randint(1, 1000, size=(2, 2), device=devices[0])

    out1_ref = model(inp1, use_cache=True, output_hidden_states=True)
    out2_ref = model(inp2, use_cache=True, past_key_values=out1_ref.past_key_values)
    out3_ref = model(inp3, use_cache=True, past_key_values=out2_ref.past_key_values)

    tp_config = None
    if use_config:
        tp_config = get_bloom_config(model_config, devices)
    model_tp = TensorParallel(model, devices, config=tp_config)
    del model

    out1 = model_tp(inp1, use_cache=True, output_hidden_states=True)
    out2 = model_tp(inp2, use_cache=True, past_key_values=out1.past_key_values)
    out3 = model_tp(inp3, use_cache=True, past_key_values=out2.past_key_values)

    assert torch.allclose(out1_ref.hidden_states[-1], out1.hidden_states[-1], atol=3e-3)
    assert torch.allclose(out1_ref.logits, out1.logits, atol=3e-3)
    assert torch.allclose(out2_ref.logits, out2.logits, atol=3e-3)
    assert torch.allclose(out3_ref.logits, out3.logits, atol=3e-3)


@pytest.mark.parametrize("num_beams", [1, 3])
def test_bloom_generate(num_beams, model_name="bigscience/bloom-560m"):
    devices=["cpu"] * 2
    model_config = transformers.AutoConfig.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).float().to(devices[0])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "Hello there!"

    gen_ref = tokenizer.decode(
        model.generate(
            tokenizer([prompt], return_tensors='pt')["input_ids"].to(devices[0]),
            num_beams=num_beams
        )[0]
    )

    tp_config = get_bloom_config(model_config, devices)
    model_tp = TensorParallelPreTrainedModel(model, devices, config=tp_config)
    del model

    gen = tokenizer.decode(
        model_tp.generate(
            tokenizer([prompt], return_tensors='pt')["input_ids"].to(devices[0]),
            num_beams=num_beams
        )[0]
    )

    assert gen == gen_ref
