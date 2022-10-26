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

def update_results(model_type, config, batch_size, 
                   seq_length, do_backward, rank, 
                   mean_iter_time, benchmark_results: dict,
                   path) -> None:

    benchmark_results["model_type"].append(model_type)
    benchmark_results["config"].append(config)
    benchmark_results["batch_size"].append(batch_size)
    benchmark_results["seq_length"].append(seq_length)
    benchmark_results["do_backward"].append(do_backward)
    benchmark_results["rank"].append(rank)
    benchmark_results["mean_iter_time"].append(mean_iter_time)

    pd.DataFrame(benchmark_results).to_csv(path)
    return None


benchmark_results = {
    "model_type":[],
    "config":[],
    "batch_size":[],
    "seq_length":[],
    "do_backward":[],
    "rank":[],
    "mean_iter_time":[],
}


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

NUM_ITER = 100

path = "results/markbench_2_gpu_results.csv"
#####################################################################

def run_experiment(rank, size):
    torch.manual_seed(1234)
    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")

    for batch_size, seq_length in batch_x_seqlen:
        for model_type in model_types:
            for conf in configs:
                try:
                # if True:
                    # Data generating process. 
                    config = BloomConfig()
                    config = config.from_pretrained(f"bigscience/{conf}")

                    data   = torch.randn(batch_size, seq_length, config.hidden_size).to(device)
                    data.requires_grad = True
                    target = torch.rand((batch_size, seq_length, config.hidden_size)).to(device)

                    # Defining the model.
                    if model_type == "mlp":
                        model = ParallelMLP(config).to(device)
                    elif model_type == "attention":
                        model = ParallelAttention(config.hidden_size, 
                                                  config.n_head, 
                                                  config.hidden_dropout).to(device)
                    elif model_type == "full_block":
                        model = ParallelBlock(config).to(device)

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
                            if model_type == "full_block":
                                output = model(data)
                            else:
                                output = model(data, data)  # whats wrong with the signature

                        if torch.cuda.is_available():
                            torch.cuda.synchronize(device)
                        if torch.distributed.get_world_size() > 1:
                            torch.distributed.barrier()

                        # The very benchmark itself
                        start_time = time.perf_counter_ns()
                        for _ in range(NUM_ITER):
                            if model_type == "full_block":
                                output = model(data)
                            else:
                                output = model(data, data)  # whats wrong with the signature

                            loss = torch.nn.functional.cross_entropy(output, target)
                            epoch_loss += loss.item()

                        if torch.cuda.is_available():
                            torch.cuda.synchronize(device)

                        working_time = time.perf_counter_ns() - start_time
                        mean_iter_time = working_time / 1e6 / NUM_ITER

                        update_results(model_type, conf, batch_size, 
                                       seq_length, False, rank, 
                                       mean_iter_time, benchmark_results,
                                       path)

                        print((model_type, conf, batch_size, seq_length, False, rank, mean_iter_time))
                except:
                    print(f"Experiment failed:(no backward), batch: {batch_size}, seq: {seq_length}")
                    update_results(model_type, conf, batch_size, 
                                    seq_length, False, rank, 
                                    "Failed", benchmark_results,
                                    path)

                # Let all processes wait for eachother once again
                if torch.distributed.get_world_size() > 1:
                    torch.distributed.barrier()

                # Now let's allow backward pass.
                # Warm-up iterations
                try:
                # if True:
                    for _ in range(100):
                        if model_type == "full_block":
                            output = model(data)
                        else:
                            output = model(data, data)  # whats wrong with the signature

                    if torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                    if torch.distributed.get_world_size() > 1:
                        torch.distributed.barrier()
                    
                    # The very benchmark itself
                    start_time = time.perf_counter_ns()
                    for _ in range(NUM_ITER):
                        if model_type == "full_block":
                            output = model(data)
                        else:
                            output = model(data, data)  # whats wrong with the signature)

                        loss = torch.nn.functional.cross_entropy(output, target)
                        epoch_loss += loss.item()
                        loss.backward()
                        optimizer.step()

                    if torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                    
                    working_time = time.perf_counter_ns() - start_time
                    mean_iter_time = working_time / 1e6 / NUM_ITER

                    update_results(model_type, conf, batch_size, 
                                    seq_length, True, rank, 
                                    mean_iter_time, benchmark_results,
                                    path)

                    print((model_type, conf, batch_size, seq_length, False, rank, mean_iter_time))
                except:
                    print(f"Experiment failed: (with backward), batch: {batch_size}, seq: {seq_length}")
                    update_results(model_type, conf, batch_size, 
                                    seq_length, True, rank, 
                                    "Failed", benchmark_results,
                                    path)


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=run_experiment, backend=BACKEND)
