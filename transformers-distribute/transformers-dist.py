import os

rank = os.environ["RANK"]
local_rank = os.environ["LOCAL_RANK"]
world_size = os.environ["WORLD_SIZE"]
print(f"rank={rank}, local_rank={local_rank}, world_size={world_size}")

import torch.distributed as dist
dist.init_process_group(backend="nccl")
print("connected")



