import torch
import os

from transformers.models.bloom.modeling_bloom import BloomModel
from transformers.models.bert.modeling_bert import BertModel
NAME = "bigscience/bloom-560m" # "bert-base-uncased"

from slicer_wrapper import SlicingConfig, tensor_parallel
from slicing_configs import SLICING_CONFIGS

import torch.distributed as dist

BACKEND = 'nccl' if torch.cuda.is_available() else 'gloo'
torch.set_num_threads(1)


def init_process(local_rank, fn, backend=BACKEND):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)

def converter_main(rank, size):
    test_input = torch.tensor([[1, 2, 3, 4, 5]]).to(f"cuda:{rank}")

    if rank == 0:
        print(f"Rank {rank} SINGLE GPU forward")
        model = BloomModel.from_pretrained(NAME).to("cuda:0")
        try:
            target_output = model(test_input).last_hidden_state
        except:
            pass

    print(f"Rank {rank} loading parallel")
    model = tensor_parallel(BloomModel, SLICING_CONFIGS[NAME], rank=rank, world_size=size).from_pretrained(NAME)
    model = model.to(f"cuda:{rank}")
    
    try:
        sharded_output = model(test_input).last_hidden_state
    except:
        pass

    if rank == 0:
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
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=converter_main, backend=BACKEND)
