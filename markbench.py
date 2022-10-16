import os
import time

import pandas as pd
import torch
import torch.distributed as dist
from   transformers.models.bloom.configuration_bloom import BloomConfig

from parallel_blocks  import ParallelBlock, ParallelMLP, ParallelAttention
from experiment_utils import update_results


# Parallel settings ###########################################
BACKEND = 'nccl' if torch.cuda.is_available() else 'gloo'
torch.set_num_threads(1)

def init_process(local_rank, fn, backend=BACKEND):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)


# benchmark_results = pd.DataFrame()


# experiment setting. ###############################################
model_types    = ["mlp", "attention", "full_block"]

configs        = ["bloom-560m",
                  "bloom-1b1", "bloom-1b7",
                  "bloom-3b", 
                  "bloom-7b1", 
                  "bloom"]  # 176B parameters.

batch_x_seqlen = [(1, 1), (1, 128), (1, 512), (1, 2048),
                  (4, 32), (4, 64),
                  (8, 64), 
                  (16, 64)]

#####################################################################

def run_experiment(rank, size):
    torch.manual_seed(1234)
    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")

    for batch_size, seq_length in batch_x_seqlen:
        for model_type in model_types:
            for conf in configs:
                try:
                    # Data generating process. 
                    config = BloomConfig()
                    config = config.from_pretrained(f"bigscience/{conf}")

                    data   = torch.randn(batch_size, seq_length, config.hidden_size).to(device)
                    data.requires_grad = True
                    target = torch.rand((batch_size, seq_length, config.hidden_size)).to(device)

                    # Defining the model.
                    if model_type == "mlp":
                        data = (data, data)
                        model = ParallelMLP(config).to(device)
                    elif model_type == "attention":
                        model = ParallelAttention(congig).to(device)
                    elif model_type == "full_block":
                        model = ParallelBlock(congig).to(device)

                    for param in model.parameters():
                        param.requires_grad = False

                    optimizer = torch.optim.SGD(model.parameters(),
                                                lr=0.01, momentum=0.5)
                    steps, epoch_loss = 0, 0

                    # Let all processes wait for eachother
                    if torch.distributed.get_world_size() > 1:
                        torch.distributed.barrier()

                    # Run with no backward passes
                    with torch.no_grad():

                        # Warm-up iterations              
                        for _ in range(100):
                            output = model(data)

                        if torch.cuda.is_available():
                            torch.cuda.synchronize(device)
                        if torch.distributed.get_world_size() > 1:
                            torch.distributed.barrier()

                        # The very benchmark itself
                        start_time = time.perf_counter_ns()
                        for _ in range(NUM_ITER):
                            output = model(data)
                            loss = torch.nn.functional.cross_entropy(output, target)
                            epoch_loss += loss.item()

                        if torch.cuda.is_available():
                            torch.cuda.synchronize(device)

                        working_time = time.perf_counter_ns() - start_time
                        
                        print(f"Rank {rank}, no backward, batch_size {batch_size}, seq_len {seq_length}; ")
                        print(f"Model type {model_type}, config: {conf}; ")
                        print(f"Mean Iter time: {working_time / 1e6 / NUM_ITER:,.2f} ms; \n")
                    except:
                        print("Experiment failed for some reason (no backward)")

                # Let all processes wait for eachother once again
                if torch.distributed.get_world_size() > 1:
                    torch.distributed.barrier()

                # Now let's allow backward pass.
                # Warm-up iterations
                try:
                    for _ in range(100):
                        output = model(data)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                    if torch.distributed.get_world_size() > 1:
                        torch.distributed.barrier()
                    
                    # The very benchmark itself
                    start_time = time.perf_counter_ns()
                    for _ in range(NUM_ITER):
                        output = model(data)
                        loss = torch.nn.functional.cross_entropy(output, target)
                        epoch_loss += loss.item()
                        loss.backward()
                        optimizer.step()

                    if torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                    
                    working_time = time.perf_counter_ns() - start_time
                    
                    print(f"Rank {rank}, with backward, batch_size {batch_size}, seq_len {seq_length}; ")
                    print(f"Model type {model_type}, config: {conf}; ")
                    print(f"Mean Iter time: {working_time / 1e6 / NUM_ITER:,.2f} ms; \n")
                except:
                    print("Experiment failed for some reason (with backward)")


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=run_experiment, backend=BACKEND)
