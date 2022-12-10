import torch

from transformers.models.bloom.modeling_bloom import BloomModel
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.t5.modeling_t5 import T5Model
NAME = "bigscience/bloom-560m" # "t5-small" # "bert-base-uncased"
MODEL_CLS = BloomModel

from transformers import logging
logging.set_verbosity_error()


from slicer_wrapper_interface import tensor_parallel

def converter_main():
    test_input = torch.tensor([[1, 2, 3, 4, 5]])

    print(f"Whole model forward")
    model = MODEL_CLS.from_pretrained(NAME)

    print(f"Whole model is loaded")
    target_output = model(test_input).last_hidden_state

    print(f"Loading slices")
    model = tensor_parallel(MODEL_CLS, devices=["cpu", "cpu"]).from_pretrained(NAME)
    
    print(f"Slices forward")
    sharded_output = model(test_input).last_hidden_state.cpu()

    print(f"Asserting allclose")
    # print("Sharded:", sharded_output[:10][:10])
    # print("CPU:", cpu_output[:10][:10])

    adiff = float((target_output - sharded_output).abs().mean())
    print("Mean absolute difference:", adiff)

    rdiff = float(((target_output - sharded_output).abs() / target_output.abs()).mean())
    print("Mean relative difference:", rdiff)

    assert torch.allclose(target_output, sharded_output, rtol=1e-3, atol=1e-3)
    print("Allclose indeed")


if __name__ == "__main__":
    converter_main()
