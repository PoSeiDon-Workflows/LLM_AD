# %% [markdown]
# # Online detection with LLMs
#
# * Follow the [README.md](../README.md) to set up the environment and data sources.
# * Dataset: `1000 Genome` (binary labels)
# * Pretrained model: `bert-base-uncased`
# %%
import logging
import pickle
import sys
from collections import defaultdict
from datetime import datetime

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from matplotlib import tight_layout
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          pipeline)

sys.path.append('../')

from data_processing import build_text_data, load_tabular_data
from utils import create_dir

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
logging.getLogger("transformers").setLevel(logging.CRITICAL)


# %%
name = "1000genome"
raw_dataset = load_dataset("csv",
                           data_files={"train": f"../data/{name}/train.csv",
                                       "validation": f"../data/{name}/validation.csv",
                                       "test": f"../data/{name}/test.csv"})

ckps = [
    "albert-base-v2",
    "albert-large-v2",
    "bert-base-cased",
    "bert-base-uncased",
    "bert-large-cased",
    "bert-large-uncased",
    "distilbert-base-cased",
    "distilbert-base-uncased",
    "roberta-base",
    "roberta-large",
    "xlnet-base-cased",
    "xlnet-large-cased"
]

metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

ckp = "bert-base-uncased"
ckp_path = f"../models/1000genome/{ckp}"
# %%
zsc = pipeline("zero-shot-classification", model=ckp_path, device=0)
text = raw_dataset['test'][0]
count = 0
first_count = []
res = zsc(text['text'], ['normal', 'anomalous'])
scores = res['scores']
label = text['label']
print(res)
substrings = text['text'].split(" ")
if scores[label] > scores[1 - label]:
    # count += 1
    substrings = text['text'].split(" ")
    pairs = [" ".join(substrings[i:i + 3]) for i in range(0, len(substrings), 3)]
    for j in range(1, len(pairs)):
        res_j = zsc(" ".join(pairs[:j]), ['normal', 'anomalous'])
        # scores_j = res_j['scores']
        # if scores_j[label] > scores_j[1 - label]:
        #     # print(f"first {j}")
        #     first_count.append(j)
        #     break

# %%

zsc = pipeline("zero-shot-classification", model=ckp_path, device=0)
test_dataset = raw_dataset['test']
count = 0
first_count = []
for i in tqdm(range(len(test_dataset))):
    res = zsc(test_dataset[i]['text'], ['normal', 'anomalous'])
    scores = res['scores']
    label = test_dataset[i]['label']
    if scores[label] > scores[1 - label]:
        count += 1
        substrings = test_dataset[i]['text'].split(" ")
        pairs = [" ".join(substrings[i:i + 3]) for i in range(0, len(substrings), 3)]
        for j in range(1, len(pairs)):
            res_j = zsc(" ".join(pairs[:j]), ['normal', 'anomalous'])
            scores_j = res_j['scores']
            if scores_j[label] > scores_j[1 - label]:
                # print(f"first {j}")
                first_count.append(j)
                break

# %%
clf = pipeline("text-classification",
               model=ckp_path,
               tokenizer=ckp_path,
               device=0
               )
test_dataset = raw_dataset['test']
count = 0
first_count = []
for i in tqdm(range(len(test_dataset))):
    res = clf(test_dataset[i]['text'])[0]
    scores = [0, 0]
    label = int(res['label'].split("_")[1])
    true_y = test_dataset[i]['label']
    scores[label], scores[1 - label] = res['score'], 1 - res['score']
    if label == true_y:
        count += 1
        substrings = test_dataset[i]['text'].split(" ")
        pairs = [" ".join(substrings[i:i + 3]) for i in range(0, len(substrings), 3)]
        for j in range(1, len(pairs)):
            res_j = clf(" ".join(pairs[:j]))[0]
            scores_j = [0, 0]
            label_j = int(res_j['label'].split("_")[1])
            scores_j[label_j], scores_j[1 - label_j] = res_j['score'], 1 - res_j['score']
            if label_j == true_y:
                # print(f"first {j}")
                first_count.append(j)
                break


# %%
from collections import Counter

import matplotlib.pyplot as plt

count_dict = Counter(first_count)
STATES = [
    'wms_delay',
    'queue_delay',
    'runtime',
    'post_process',
    'stage_in_delay',
    'stage_out_delay',
    "bytes_in",
    "bytes_out",
    "cpu_time"]
fig = plt.figure(figsize=(4, 3), tight_layout=True, dpi=600)
plt.bar(count_dict.keys(), count_dict.values())
# plt.xlabel("Job states")
plt.ylim(0, 3000)
# plt.yscale("log")
plt.ylabel("Number of samples", fontsize=8)
plt.tick_params(axis='y', labelsize=8)
plt.xticks(range(1, 10), STATES, rotation=45, ha="right", fontsize=8)

for i, v in count_dict.items():
    plt.text(i, v + 8, str(v), color='black', fontsize=8, ha="center")
plt.savefig("online_detection.pdf")
# %%


# %%
# ckp_path = "bert-base-uncased"
clf = pipeline("text-classification",
               model=ckp_path,
               tokenizer=ckp_path,
               device=0
               )
# test_dataset = raw_dataset['test']
text = raw_dataset['test'][3]
res = clf(text['text'])[0]
scores = [0, 0]
label = int(res['label'].split("_")[1])
true_y = text['label']
scores[label], scores[1 - label] = res['score'], 1 - res['score']

substrings = text['text'].split(" ")
pairs = [" ".join(substrings[i:i + 3]) for i in range(0, len(substrings), 3)]
for j in range(1, len(pairs)):
    print(" ".join(pairs[:j]))
    res_j = clf(" ".join(pairs[:j]))[0]
    scores_j = [0, 0]

    print(f"==> label: {res_j['label']}, score: {res_j['score']:.4f}")

# %%
