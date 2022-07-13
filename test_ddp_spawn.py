import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def example(local_rank, rank, nproc_per_node, world_size):
    rank = rank*nproc_per_node + local_rank
    print("local_rank:", local_rank, "global_rank:", rank, "world_size:", world_size)
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # create local model
    #model = nn.Linear(10, 10).to(rank)
    model = nn.Linear(10, 10)
    # construct DDP model
    #ddp_model = DDP(model, device_ids=[rank])
    ddp_model = DDP(model)
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    #outputs = ddp_model(torch.randn(20, 10).to(rank))
    outputs = ddp_model(torch.randn(20, 10))
    #labels = torch.randn(20, 10).to(rank)
    labels = torch.randn(20, 10)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int)
    args = parser.parse_args()
    rank = args.rank
    nodes = 2
    nproc_per_node = 2
    world_size = nodes * nproc_per_node
    mp.spawn(example,
        args=(rank, nproc_per_node, world_size),
        nprocs=nproc_per_node,
        join=True)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
