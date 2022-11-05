import os
import torch
import torch.distributed as dist

from transformers.models.bloom.modeling_bloom import BloomModel, BloomBlock
from transformers.models.bloom.configuration_bloom import BloomConfig

from parallel_blocks import ParallelBlock

# Model Selection
NAME = "bigscience/bloom-560m"

# Parallel settings ###########################################
BACKEND = 'nccl' if torch.cuda.is_available() else 'gloo'
torch.set_num_threads(1)


def init_process(local_rank, fn, backend=BACKEND):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)


def convert_block_parallel(block: BloomBlock, rank: int, world_size: int, config: BloomConfig):
    parallel_block = ParallelBlock(config)

    cut_size = 4 * config.hidden_size // world_size
    parallel_block.mlp.dense_h_to_4h.weight.data = block.mlp.dense_h_to_4h.weight.data[:, rank * cut_size: (rank + 1) * cut_size].clone().detach().contiguous()
    cut_size = 4 * config.hidden_size // world_size
    parallel_block.mlp.dense_4h_to_h.weight.data = block.mlp.dense_4h_to_h.weight.data[rank * cut_size: (rank + 1) * cut_size, :].clone().detach().contiguous()

    cut_size = 3 * config.hidden_size // world_size
    parallel_block.self_attention.query_key_value.weight.data = block.self_attention.query_key_value.weight.data[:, rank * cut_size: (rank + 1) * cut_size].clone().detach().contiguous()
    cut_size = config.hidden_size // world_size
    parallel_block.self_attention.dense.weight.data = block.self_attention.dense.weight.data[rank * cut_size: (rank + 1) * cut_size, :].clone().detach().contiguous()

    return parallel_block


def convert_bloom_parallel(model: BloomModel, rank: int, world_size: int, config: BloomConfig):
    for block in model.h:
        block = convert_block_parallel(block, rank, world_size, config)

    return model

def convert_and_scatter_bloom_parallel(model: BloomModel, rank: int, world_size: int, config: BloomConfig):
    sharded_model = convert_bloom_parallel(model, rank, world_size, BloomConfig.from_pretrained(NAME))
    sharded_model = sharded_model.to(f"cuda:{rank}")
    return sharded_model


def converter_main(rank, size):
    test_input = torch.tensor([[1, 2, 3, 4, 5]])

    print(f"Rank {rank} loading")
    model = BloomModel.from_pretrained(NAME)
    if rank == 0:
        print(f"Rank {rank} CPU forward")
        cpu_output = model(test_input).last_hidden_state

    print(f"Rank {rank} slicing")
    sharded_model = convert_and_scatter_bloom_parallel(model, rank, size, BloomConfig.from_pretrained(NAME))
    print(f"Rank {rank } testing forward")
    sharded_output = sharded_model(test_input.to(f"cuda:{rank}")).last_hidden_state

    if rank == 0:
        print(f"Asserting allclose")
        # print("Sharded:", sharded_output[:10][:10])
        # print("CPU:", cpu_output[:10][:10])

        adiff = float((cpu_output - sharded_output.cpu()).abs().mean())
        print("Mean absolute difference:", adiff)

        rdiff = float(((cpu_output - sharded_output.cpu()).abs() / cpu_output.abs()).mean())
        print("Mean relative difference:", rdiff)

        assert torch.allclose(cpu_output, sharded_output.cpu(), rtol=1e-3, atol=1e-3)
        print("Allclose indeed")


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=converter_main, backend=BACKEND)
