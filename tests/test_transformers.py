from typing import Sequence

import pytest
import torch
import transformers
from transformers import AutoTokenizer

from tensor_parallel import TensorParallel, tensor_parallel
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
@pytest.mark.parametrize("model_name", ["t5-small"])  # "bigscience/bloom-560m"
def test_generate(num_beams, model_name):
    def _generate_scores(model, tokenizer, prompt, num_beams):
        return model.generate(
            tokenizer([prompt], return_tensors="pt")["input_ids"].to(devices[0]),
            num_beams=num_beams,
            min_length=5,
            return_dict_in_generate=True,
            output_scores=True,
        ).scores[0]

    def _get_scores_allclose_length(first_scores: Sequence[torch.Tensor], second_scores: Sequence[torch.Tensor]) -> int:
        length = 0
        while (
            length < len(first_scores)
            and length < len(second_scores)
            and torch.allclose(first_scores[length], second_scores[length], atol=3e-3, rtol=3e-3)
        ):
            length += 1
        return length

    devices = ["cpu"] * 2
    if model_name == "t5-small":
        model = (
            transformers.T5ForConditionalGeneration.from_pretrained(model_name, low_cpu_mem_usage=True)
            .float()
            .to(devices[0])
        )
    else:
        model = (
            transformers.AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).float().to(devices[0])
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt = "Translate from German to English: How are you?"

    scores_ref = _generate_scores(model, tokenizer, prompt, num_beams)

    model_tp = tensor_parallel(model, devices)
    del model

    scores = _generate_scores(model_tp, tokenizer, prompt, num_beams)

    matching_len = _get_scores_allclose_length(scores_ref, scores)
    assert matching_len > 3, ".generate() diverges too quickly"
