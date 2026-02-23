# accelerate config 添加配置
# accelerate launch [--config_file <配置文件路径>] xx.py
import os
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset

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

datasets = load_dataset("csv", data_files="/root/autodl-tmp/code/test-ddp/ChnSentiCorp_htl_all.csv", split="train")
datasets = datasets.filter(lambda x: x["review"] is not None)
datasets = datasets.train_test_split(test_size=0.1, seed=42)

import torch

device = torch.device(f"cuda:{local_rank}")

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

def process_function(examples):
    tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples

tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)
print(tokenized_datasets)

model = BertForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", device_map={"": local_rank}, low_cpu_mem_usage=True)

import evaluate

acc_metric = evaluate.load("accuracy")
f1_metirc = evaluate.load("f1")

def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metirc.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc

from torch.distributed import ReduceOp

from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

train_args = TrainingArguments(output_dir="/root/autodl-tmp/code/test-ddp/checkpoints",      # 输出文件夹
                               per_device_train_batch_size=32,  # 训练时的batch_size
                               per_device_eval_batch_size=128,  # 验证时的batch_size
                               logging_steps=10,                # log 打印的频率
                               eval_strategy="epoch",     # 评估策略
                               save_strategy="epoch",           # 保存策略
                               save_total_limit=3,              # 最大保存数
                               learning_rate=2e-5,              # 学习率
                               weight_decay=0.01,               # weight_decay
                               metric_for_best_model="f1",      # 设定评估指标
                               report_to=["tensorboard"],
                               load_best_model_at_end=True)     # 训练完成后加载最优模型

trainer = Trainer(model=model, 
                  args=train_args, 
                  train_dataset=tokenized_datasets["train"], 
                  eval_dataset=tokenized_datasets["test"], 
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=eval_metric)

trainer.train()

# dist.destroy_process_group()
# print(f"{rank}: Training finished and process group destroyed.")