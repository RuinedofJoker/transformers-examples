"""
CUDA_BISIBLE_DEVICES=0,1 \
torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  --node_rank=0 \
  --rdzv_id=123 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29500 \
ddp.py

CUDA_BISIBLE_DEVICES=2,3 \
torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  --node_rank=1 \
  --rdzv_id=123 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29500 \
ddp.py
"""

import os
from transformers import BertTokenizer, BertForSequenceClassification

os.environ["http_proxy"] = "http://127.0.0.1:10808"
os.environ["https_proxy"] = "http://127.0.0.1:10808"
os.environ["HF_HOME"] = "/root/autodl-tmp/HF_download"
os.environ["MODELSCOPE_CACHE"] = "/root/autodl-tmp/MODELSCOPE_download"
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.environ["TRANSFORMERS_NO_ADVISORY"] = "1"

import torch.distributed as dist
dist.init_process_group(backend="nccl")

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

import pandas as pd
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("/root/autodl-tmp/code/test-ddp/ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
    
    def __len__(self):
        return len(self.data)

dataset = MyDataset()

import torch
from torch.utils.data import random_split
trainset, validset = random_split(dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42))

device = torch.device(f"cuda:{local_rank}")

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels)
    return inputs

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_func, sampler=DistributedSampler(trainset))
validloader = DataLoader(validset, batch_size=64, collate_fn=collate_func, sampler=DistributedSampler(validset))

from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP

model = BertForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", device_map={"": local_rank}, low_cpu_mem_usage=True)
model = DDP(model)
optimizer = Adam(model.parameters(), lr=2e-5)

def print_rank_0(info):
  if rank == 0:
    print(info)

def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = batch.to(local_rank)
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch["labels"].long()).float().sum()
    dist.all_reduce(acc_num)
    return acc_num / len(validset)

from torch.distributed import ReduceOp

def train(epoch=3, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        trainloader.sampler.set_epoch(ep)
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = batch.to(local_rank)
            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                dist.all_reduce(loss, op=ReduceOp.AVG)
                print_rank_0(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
            global_step += 1
        acc = evaluate()
        print_rank_0(f"ep: {ep}, acc: {acc}")

train()

dist.destroy_process_group()
print(f"{rank} Training finished and process group destroyed.")