import os
import time

import torch
from torch import nn
from   transformers.models.bloom.configuration_bloom import BloomConfig
from pipelineable_blocks import MiddleBloom

from torchgpipe import GPipe

    
# Arg parse ################################################### 
import getopt, sys

# default values.
TRAIN:       bool = False
NUM_ITER:    int  = 100
BATCH_SIZE:  int  = 128
SEQ_LENGTH:  int  = 17
BLOOMCONFIG: str  = "bloom-7b1"
DEPTH:       int  = 2
HALF:        bool = False

argumentList = sys.argv[1:]
options = "t:n:b:s:c:d:p:h:"
long_options = ["train", "num_iter=",
                "batch_size=", "seq_length=", 
                "bloomconfig=", "depth=", "half"]

arguments, values = getopt.getopt(argumentList, options, long_options)
for currentArgument, currentValue in arguments:
        if currentArgument in ("-t", "--train"):
            TRAIN = True      
        elif currentArgument in ("-n", "--num_iter"):
            NUM_ITER = int(currentValue)
        elif currentArgument in ("-b", "--batch_size"):
            BATCH_SIZE = int(currentValue)
        elif currentArgument in ("-s", "--seq_length"):
            SEQ_LENGTH = int(currentValue)
        elif currentArgument in ("-c", "--bloomconfig"):
            BLOOMCONFIG = currentValue
        elif currentArgument in ("-d", "--depth"):
            DEPTH = int(currentValue)
        elif currentArgument in ("-h", "--half"):
            HALF = True

print(f"Benchmark setting:")
print(f"TRAIN: {TRAIN}")
print(f"NUM_ITER:    {NUM_ITER}")
print(f"BATCH_SIZE:  {BATCH_SIZE}")
print(f"SEQ_LENGTH:  {SEQ_LENGTH}")
print(f"BLOOMCONFIG: {BLOOMCONFIG}")
print(f"DEPTH: {DEPTH}")

##############################################################

def run_training():
    torch.manual_seed(1234)

    # Config #############################################
    config = BloomConfig()
    config = config.from_pretrained(f"bigscience/{BLOOMCONFIG}")
    ######################################################

    # Your model here. ###################################
    available_gpus = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
    assert DEPTH % len(available_gpus) == 0
    config.num_hidden_layers = DEPTH // len(available_gpus)

    if len(available_gpus) == 1:
        device = 'cuda:0'
        model = MiddleBloom(config).to(device)
        in_device = device
        out_device = device
    else:
        model = GPipe(nn.Sequential(*[MiddleBloom(config) for _ in available_gpus]),
                    balance=[1 for _ in available_gpus],
                    devices=available_gpus,
                    chunks=BATCH_SIZE // 8)

        in_device = model.devices[0]
        out_device = model.devices[-1]

    if HALF:
        model = model.half()
    ######################################################

    # Data generating process. ###########################
    data = torch.randn(BATCH_SIZE, SEQ_LENGTH, config.hidden_size).to(in_device)
    target = torch.rand((BATCH_SIZE, SEQ_LENGTH, config.hidden_size)).to(out_device)

    if HALF:
        data = data.half()
        target = target.half()
    ######################################################

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    steps = 0
    epoch_loss = 0
    # TRAIN = False  # why do we need that crunch?
    with torch.cuda.amp.autocast():
        if TRAIN:
            start_time = time.perf_counter_ns()
           
            # Warm-up iterations.
            for iter in range(max(10, NUM_ITER // 10)):
                 output = model(data)

            if torch.cuda.is_available():
                 torch.cuda.synchronize(out_device)

            for iter in range(NUM_ITER):

                optimizer.zero_grad()
                output = model(data)
                loss = torch.nn.functional.mse_loss(output, target)
                epoch_loss += loss.item()
                loss.backward()

                optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize(out_device)

            working_time = time.perf_counter_ns() - start_time
        else:
            with torch.no_grad():
                # Warm-up iterations.               
                for iter in range(max(10, NUM_ITER // 10)):
                    output = model(data)

                if torch.cuda.is_available():
                    torch.cuda.synchronize(out_device)

                start_time = time.perf_counter_ns()
                for iter in range(NUM_ITER):
                    output = model(data)

                    loss = torch.nn.functional.cross_entropy(output, target)
                    epoch_loss += loss.item()

                if torch.cuda.is_available():
                    torch.cuda.synchronize(out_device)

                working_time = time.perf_counter_ns() - start_time

    print(f"Mean Iter: {working_time / 1e6 / NUM_ITER:,.2f} ms; ")

if __name__ == "__main__":
    run_training()
