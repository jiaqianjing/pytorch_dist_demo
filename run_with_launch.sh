# node 0 (172.17.0.3)
python -m torch.distributed.launch --nnode 2 --nproc_per_node 2 --node_rank 0 --master_addr 172.17.0.3 --master_port 25900 test_ddp_launch.py

# node 1 (172.17.0.2)
# python -m torch.distributed.launch --nnode 2 --nproc_per_node 2 --node_rank 1 --master_addr 172.17.0.3 --master_port 25900 test_ddp_launch.py
