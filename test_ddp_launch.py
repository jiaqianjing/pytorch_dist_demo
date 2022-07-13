import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt/nprocs
    return rt

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def main():
    # nccl if you want to use gpu
    dist.init_process_group('gloo')
    rank = dist.get_rank()
    print("----> global rank:", rank)
    # refer to gpu device
    # deivce_id = os.environ['LOCAL_RANK']
    # model = ToyModel().to(device_id)
    # ddp_model = DDP(model, device_ids=[device_id])

    model = ToyModel()
    ddp_model = DDP(model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    # labels = torch.randn(20, 5).to(device_id)
    labels = torch.randn(20, 5)
    loss = loss_fn(outputs, labels)
    print(f"rank {rank}, loss: {loss}")
    loss.backward()
    optimizer.step()
    avg_loss = reduce_mean(loss, dist.get_world_size())
    if rank == 0:
        print("all nodes mean loss:", avg_loss.detach().cpu().numpy())


if __name__ == "__main__":
    print("RANK:", os.environ['RANK'])
    print("LOCAL RANK:", os.environ['LOCAL_RANK'])
    print("MASTER_ADDR:", os.environ['MASTER_ADDR'])
    print("MASTER_PORT:", os.environ['MASTER_PORT'])
    main()
