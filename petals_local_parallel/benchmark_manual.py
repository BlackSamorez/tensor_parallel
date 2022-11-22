import os
import time

import pandas as pd
import torch
import torch.distributed as dist
from   transformers.models.bloom.configuration_bloom import BloomConfig

from parallel_blocks import ParallelBlock, ParallelMLP, ParallelAttention


# Parallel settings ###########################################
BACKEND = 'nccl' if torch.cuda.is_available() else 'gloo'
torch.set_num_threads(1)

def init_process(local_rank, fn, backend=BACKEND):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)
    
# Arg parse ################################################### 
import getopt, sys

# default values.
DO_BACKWARD: int = False
NUM_ITER: int    = 100
BATCH_SIZE: int  = 4
SEQ_LENGTH: int  = 17
BLOOMCONFIG: str = "bloom"

argumentList = sys.argv[1:]
options = "d:n:b:s:c:"
long_options = ["do_backward=", "num_iter=",
                "batch_size=", "seq_length=", 
                "bloomconfig="]

arguments, values = getopt.getopt(argumentList, options, long_options)
for currentArgument, currentValue in arguments:
        if currentArgument in ("-d", "--do_backward"):
            DO_BACKWARD = int(currentValue)      
        elif currentArgument in ("-n", "--num_iter"):
            NUM_ITER = int(currentValue)
        elif currentArgument in ("-b", "--batch_size"):
            BATCH_SIZE = int(currentValue)
        elif currentArgument in ("-m", "--maximum"):
            MAXIMUM = True
        elif currentArgument in ("-s", "--seq_length"):
            SEQ_LENGTH = int(currentValue)
        elif currentArgument in ("-c", "--bloomconfig"):
            BLOOMCONFIG = currentValue

print(f"Benchmark setting:")
print(f"DO_BACKWARD: {DO_BACKWARD}")
print(f"NUM_ITER:    {NUM_ITER}")
print(f"BATCH_SIZE:  {BATCH_SIZE}")
print(f"SEQ_LENGTH:  {SEQ_LENGTH}")
print(f"BLOOMCONFIG: {BLOOMCONFIG}")

##############################################################

benchmark_results = pd.DataFrame()

def update_results(res_df: pd.DataFrame, rank: int,
                   do_backward: int,
                   num_iter: int,
                   batch_size: int,
                   seq_length: int,
                   hid_size: int,
                   model_type: str,
                   config: str,
                   device_name: str,
                   working_time) -> pd.DataFrame():

    res_df[f"{rank}_do_backward"] = do_backward
    res_df[f"{rank}_num_iter"]    = num_iter
    res_df[f"{rank}_batch_size"]  = batch_size
    res_df[f"{rank}_seq_length"]  = seq_length
    res_df[f"{rank}_config"]      = config
    res_df[f"{rank}_hid_size"]    = hid_size
    res_df[f"{rank}_model_type"]  = model_type
    res_df[f"{rank}_device_name"] = device_name

    # Store the computation time.
    res_df[f"{rank}_working_time"] = working_time

    return res_df


def run_training(rank, size):
    torch.manual_seed(1234)
    device = torch.device('cuda', rank) if torch.cuda.is_available() else torch.device('cpu')

    # Data generating process. ###########################
    config = BloomConfig()
    config = config.from_pretrained(f"bigscience/{BLOOMCONFIG}")

    data = torch.randn(BATCH_SIZE, SEQ_LENGTH, config.hidden_size).to(device)
    target = torch.rand((BATCH_SIZE, SEQ_LENGTH, config.hidden_size)).to(device)
    ######################################################


    # Your model here. ###################################
    model = ParallelMLP(config).to(device)
    ######################################################

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    steps = 0
    epoch_loss = 0
    # DO_BACKWARD = False  # why do we need that crunch?
    if torch.distributed.get_world_size() > 1:
        torch.distributed.barrier()
    with torch.cuda.amp.autocast():
        if DO_BACKWARD:
            start_time = time.perf_counter_ns()
           
            # Warm-up iterations.
            for iter in range(100):
                 output = model(data, data)

            if torch.cuda.is_available():
                 torch.cuda.synchronize(device)
            if torch.distributed.get_world_size() > 1:
                 torch.distributed.barrier()

            for iter in range(NUM_ITER):

                optimizer.zero_grad()
                output = model(data, data)
                loss = torch.nn.functional.mse_loss(output, target)
                epoch_loss += loss.item()
                loss.backward()

                optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize(device)

            working_time = time.perf_counter_ns() - start_time
        else:
            with torch.no_grad():
                # Warm-up iterations.               
                for iter in range(100):
                    output = model(data, data)

                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                if torch.distributed.get_world_size() > 1:
                    torch.distributed.barrier()

                start_time = time.perf_counter_ns()
                for iter in range(NUM_ITER):
                    output = model(data, data)

                    loss = torch.nn.functional.cross_entropy(output, target)
                    epoch_loss += loss.item()

                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)

                working_time = time.perf_counter_ns() - start_time

    print(f"Mean Iter time for Rank {rank}: {working_time / 1e6 / NUM_ITER:,.2f} ms; ")

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=run_training, backend=BACKEND)
