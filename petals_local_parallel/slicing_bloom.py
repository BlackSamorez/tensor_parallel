import os
import torch
import torch.distributed as dist

from transformers.models.bloom.modeling_bloom import BloomModel, BloomAttention, BloomMLP

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


class ParallelBloomAttention(BloomAttention):
    def __init__(self, bloom_attention: BloomAttention, rank: int, world_size: int):
        self.__dict__ = bloom_attention.__dict__.copy()
        self.rank = rank
        self.world_size = world_size

        # num heads are now per proc
        self.num_heads = self.num_heads // self.world_size

        # slicing query_key_value
        cut_size = self.query_key_value.weight.shape[0] // self.world_size
        self.query_key_value.weight.data = self.query_key_value.weight.data[rank * cut_size: (rank + 1) * cut_size, :]
        self.query_key_value.bias.data   = self.query_key_value.bias.data[rank * cut_size: (rank + 1) * cut_size]

        # slicing dense
        cut_size = self.dense.weight.shape[1] // self.world_size
        self.dense.weight.data = self.dense.weight.data[:, rank * cut_size: (rank + 1) * cut_size]
        self.dense.bias.data   = self.dense.bias.data / self.world_size

    def forward(self, *args, **kwargs):
        # slicing alibi 
        alibi = kwargs["alibi"]
        alibi_cut_size = int(alibi.shape[0]) // self.world_size
        kwargs["alibi"] = alibi[self.rank * alibi_cut_size:(self.rank + 1) * alibi_cut_size, ...]

        # scaling residual
        residual = args[1]
        residual /= self.world_size
        args = (args[0], residual)

        # calling super
        output = super().forward(*args, **kwargs)

        # reducing result
        dist.all_reduce(output[0])
        return output


class ParallelBloomMLP(BloomMLP):
    def __init__(self, bloom_mlp: BloomMLP, rank: int, world_size: int):
        self.__dict__ = bloom_mlp.__dict__.copy()
        self.rank = rank
        self.world_size = world_size
        
        # slicing dense_h_to_4h
        cut_size = self.dense_h_to_4h.weight.shape[0] // world_size
        self.dense_h_to_4h.weight.data = self.dense_h_to_4h.weight.data[rank * cut_size: (rank + 1) * cut_size, :]
        self.dense_h_to_4h.bias.data   = self.dense_h_to_4h.bias.data[rank * cut_size: (rank + 1) * cut_size]

        # slicing dense_4h_to_h
        cut_size = self.dense_4h_to_h.weight.shape[1] // world_size
        self.dense_4h_to_h.weight.data = self.dense_4h_to_h.weight.data[:, rank * cut_size: (rank + 1) * cut_size]
        self.dense_4h_to_h.bias.data   = self.dense_4h_to_h.bias.data / world_size

    def forward(self, *args, **kwargs):
        # scaling residual
        residual = args[1]
        residual /= self.world_size
        args = (args[0], residual)

        # calling super
        output = super().forward(*args, **kwargs)

        # reducing result
        dist.all_reduce(output)
        return output


def slice_bloom(model: BloomModel, rank: int, world_size: int):
    for i, block in enumerate(model.h):
        model.h[i].self_attention = ParallelBloomAttention(block.self_attention, rank, world_size)
        model.h[i].mlp = ParallelBloomMLP(block.mlp, rank, world_size)

    return model

def slice_and_scatter_bloom(model: BloomModel, rank: int, world_size: int):
    sharded_model = slice_bloom(model, rank, world_size)
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
    sharded_model = slice_and_scatter_bloom(model, rank, size)
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
