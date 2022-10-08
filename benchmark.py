%%writefile ddp_example.py
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import MNIST

from datetime import datetime


BACKEND = 'nccl' if torch.cuda.is_available() else 'gloo'

def init_process(local_rank, fn, backend=BACKEND):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)


torch.set_num_threads(1)


# example model. ############################################
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4608, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output
##############################################################


def run_training(rank, size):
    torch.manual_seed(1234)
    device = torch.device('cuda', rank) if torch.cuda.is_available() else torch.device('cpu')

    # Data generating process. ###########################
    data = torch.randn(100, 1, 28, 28).to(device)
    target = torch.randint(0, 10, (100,)).to(device)
    ######################################################


    # Your model here. ###################################
    model = Net().to(device)
    model = DistributedDataParallel(model)
    ######################################################

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    steps = 0
    epoch_loss = 0

    start_time = datetime.now()
    for iter in range(1000):

        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        epoch_loss += loss.item()
        loss.backward()

        optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    working_time = datetime.now() - start_time

    print(f"Rank {rank} worked for {working_time}")

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=run_training, backend=BACKEND)
