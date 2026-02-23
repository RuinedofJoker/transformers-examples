import os
from transformers import BertTokenizer, BertForSequenceClassification

os.environ["http_proxy"] = "http://127.0.0.1:10808"
os.environ["https_proxy"] = "http://127.0.0.1:10808"
os.environ["HF_HOME"] = "/root/autodl-tmp/HF_download"
os.environ["MODELSCOPE_CACHE"] = "/root/autodl-tmp/MODELSCOPE_download"

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
model = BertForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext")