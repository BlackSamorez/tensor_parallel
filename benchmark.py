import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST

from transformers.models.bloom.configuration_bloom import BloomConfig

from parallel_attention import ParallelBlock

BACKEND = 'nccl' if torch.cuda.is_available() else 'gloo'

def init_process(local_rank, fn, backend=BACKEND):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)

torch.set_num_threads(1)

    
# Arg parse ################################################### 
import getopt, sys

# default values.
DO_BACKWARD: int = True
NUM_ITER: int = 100
BATCH_SIZE: int = 1000

argumentList = sys.argv[1:]
options = "d:n:b:"
long_options = ["do_backward=", "num_iter=", "batch_size="]
arguments, values = getopt.getopt(argumentList, options, long_options)

for currentArgument, currentValue in arguments:
        if currentArgument in ("-d", "--do_backward"):
            DO_BACKWARD = bool(currentValue)      
        elif currentArgument in ("-n", "--num_iter"):
            NUM_ITER = int(currentValue)
        elif currentArgument in ("-b", "--batch_size"):
            BATCH_SIZE = int(currentValue)

# print(f"Benchmark settings:")
# print(f"DO_BACKWARD: {DO_BACKWARD}\nNUM_ITER: {NUM_ITER}\nBATCH_SIZE: {BATCH_SIZE}")
##############################################################

def run_training(rank, size):
    torch.manual_seed(1234)
    device = torch.device('cuda', rank) if torch.cuda.is_available() else torch.device('cpu')

    # Data generating process. ###########################
    config = BloomConfig()

    data = torch.randn(BATCH_SIZE, 17, config.hidden_size).to(device)
    target = torch.rand((BATCH_SIZE, 17, config.hidden_size)).to(device)
    ######################################################


    # Your model here. ###################################
    model = ParallelBlock(config).to(device)
    ######################################################

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    steps = 0
    epoch_loss = 0

    if DO_BACKWARD:
        start_time = time.perf_counter_ns()
        for iter in range(NUM_ITER):

            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.mse_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()

            optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        working_time = time.perf_counter_ns() - start_time
    else:
        start_time = time.perf_counter_ns()
        for iter in range(NUM_ITER):
            output = model(data)

            # Do we need loss calculation?
            loss = torch.nn.functional.cross_entropy(output, target)
            epoch_loss += loss.item()

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        working_time = time.perf_counter_ns() - start_time

    print(f"Mean Iter time for Rank {rank}: {working_time / 1e6 / NUM_ITER:,.2f} ms; ")

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=run_training, backend=BACKEND)
