# %%
import logging
import pickle
import random
from collections import defaultdict
import select
from turtle import position
import umap
import evaluate
from matplotlib import tight_layout
import numpy as np
import pandas as pd
import pip
import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          pipeline)
from tqdm import tqdm
from data_processing import build_text_data, load_tabular_data
from utils import create_dir

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.getLogger("transformers").setLevel(logging.CRITICAL)

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

# %%
"""
print(f"{'model':<25} | {'vocab size':>10} | {'num params':>10} | {'max token len':>10}")
for ckp in ckps:
    tokenizer = AutoTokenizer.from_pretrained(ckp)
    model = AutoModelForSequenceClassification.from_pretrained(ckp, num_labels=2)
    # model.logger.setLevel(logging.ERROR)
    # num_params = sum(p.numel() for p in model.parameters())
    num_params = model.num_parameters()

    max_token_len = tokenizer.model_max_length
    print(f"{ckp:<25}   {tokenizer.vocab_size:>10}   {num_params:>10}   {max_token_len:>10.2e}")
"""

# %%
""" Print the statistic of the dataset """
"""
names = [
    "1000genome_new_2022",
    "montage",
    "predict_future_sales"
]
print(f"| {'dataset':^20} | {'split':^10} | # normal | # anomaly | % anomalous |")
print(f"| {'-'*20} | {'-'*10} | {'-'*8} | {'-'*9} | {'-'*11} |")
for name in names:
    raw_dataset = load_dataset("csv",
                               data_files={"train": f"./data/{name}/train.csv",
                                           "validation": f"./data/{name}/validation.csv",
                                           "test": f"./data/{name}/test.csv"})
    # print number of label=1, number of label=0 in train test and validation
    for cat in ["train", "validation", "test"]:
        _labels = np.array(raw_dataset[cat]["label"])
        print(f"| {name:^20} | {cat:>10} | " +
              f"{(_labels==0).sum():>8} | " +
              f"{(_labels==1).sum():>9} | " +
              f"{(_labels==1).sum() / len(_labels):>11.4f} |")

    # _labels = np.array(raw_dataset["validation"]["label"])
    # print(f"| {name:^20} | validation | {(_labels==0).sum():>8} | {(_labels==1).sum():>9} | {(_labels==1).sum() / len(_labels):>11.4f} |")
    # _labels = np.array(raw_dataset["test"]["label"])
    # print(f"| {name:^20} | test       | {(_labels==0).sum():>8} | {(_labels==1).sum():>9} | {(_labels==1).sum() / len(_labels):>11.4f} |")
"""
# %%

""" Print the bias of pretrained models (with SEED)"""
"""
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print(f"| {'model':<25} | {'label 0':>10} | {'label 1':>10} | {'pred label':>10} |")
print(f"| {'-'*25} | {'-'*10} | {'-'*10} | {'-'*10} |")
for ckp in ckps:
    clf = pipeline(task="text-classification", model=ckp)
    res = clf([""])
    # print(ckp, clf(""))
    label = int(res[0]['label'][-1:])
    if label == 0:
        print(f"| {ckp:<25} | {res[0]['score']:>10.4f} | {1-res[0]['score']:>10.4f} | {label:^10} |")
    else:
        print(f"| {ckp:<25} | {1-res[0]['score']:>10.4f} | {res[0]['score']:>10.4f} | {label:^10} |")
    # print()
"""
# %%
""" Print the bias of pretrained models (without SEED, run 10 times) """

pbar = tqdm(ckps, desc="ckp")
all_res = defaultdict(list)
for ckp in pbar:
    for _ in range(10):
        clf = pipeline(task="text-classification", model=ckp)
        # check prediction with empty string
        res = clf([""])
        # print(ckp, clf(""))
        label = int(res[0]['label'][-1:])
        scores = [0, 0]
        scores[label] = res[0]['score']
        scores[1 - label] = 1 - res[0]['score']

        all_res[ckp].append(scores)

        # if label == 0:
        #     print(f"| {ckp:<25} | {res[0]['score']:>10.4f} | {1-res[0]['score']:>10.4f} | {label:^10} |")
        # else:
        #     print(f"| {ckp:<25} | {1-res[0]['score']:>10.4f} | {res[0]['score']:>10.4f} | {label:^10} |")
        pbar.set_postfix({"ckp": ckp, "score 0": scores[0], "1": scores[1]})

# %%
# print(all_res)
# pickle.dump(all_res, open("./ckp_stats.pkl", "wb"))
np.set_printoptions(precision=4)
for ckp in all_res:
    _tmp = np.array(all_res[ckp])
    print(f"{ckp:>24}, avg {_tmp.mean(axis=0)}, std {_tmp.std(axis=0)}")

# %%
""" Plot the boxplot of the biased results from pretrained models """
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(4, 3), tight_layout=True, dpi=600)
# box style
boxprops_normal = dict(linestyle='-', linewidth=1, color='b')
boxprops_abnormal = dict(linestyle='-', linewidth=1, color='r')
for i, ckp in enumerate(all_res):
    plt.boxplot(np.array(all_res[ckp])[:, 0], positions=[i - 0.11], widths=0.2, boxprops=boxprops_normal)
    plt.boxplot(np.array(all_res[ckp])[:, 1], positions=[i + 0.11], widths=0.2, boxprops=boxprops_abnormal)

plt.hlines(0.5, -0.5, len(all_res) - 0.5, colors='k', linestyles='dashed', linewidth=0.5, label="0.5")
plt.xticks(np.arange(len(all_res)), all_res.keys(), rotation=45, ha="right", fontsize=8)
# plt.ylim(0, 1)
plt.ylabel("Classification score", fontsize=8)
# plt.xlabel("Pretrained model", fontsize=8)
plt.plot([], [], 'b-', label="normal")
plt.plot([], [], 'r-', label="abnormal")
plt.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=8)

# %%
pbar = tqdm(ckps, desc="ckp")
all_res_sft = defaultdict(list)
for ckp in pbar:
    clf = pipeline(task="text-classification", model=f"./models/1000genome_new_2022/{ckp}-sft")
    # check prediction with empty string
    res = clf([""])
    # print(ckp, clf(""))
    label = int(res[0]['label'][-1:])
    scores = [0, 0]
    scores[label] = res[0]['score']
    scores[1 - label] = 1 - res[0]['score']

    all_res_sft[ckp].append(scores)

    # if label == 0:
    #     print(f"| {ckp:<25} | {res[0]['score']:>10.4f} | {1-res[0]['score']:>10.4f} | {label:^10} |")
    # else:
    #     print(f"| {ckp:<25} | {1-res[0]['score']:>10.4f} | {res[0]['score']:>10.4f} | {label:^10} |")
    pbar.set_postfix({"ckp": ckp, "score 0": scores[0], "1": scores[1]})

# %%
all_res_sft
# %%

fig = plt.figure(figsize=(4, 3), tight_layout=True, dpi=600)
# box style
boxprops_normal = dict(linestyle='-', linewidth=1, color='b')
boxprops_abnormal = dict(linestyle='-', linewidth=1, color='r')
for i, ckp in enumerate(all_res):
    plt.boxplot(np.array(all_res[ckp])[:, 0], positions=[i - 0.11], widths=0.2, boxprops=boxprops_normal)
    plt.boxplot(np.array(all_res[ckp])[:, 1], positions=[i + 0.11], widths=0.2, boxprops=boxprops_abnormal)
    # plt.scatter(i - 0.11, np.array(all_res_sft[ckp])[:, 0], s=2, c='b', marker='x')
    # plt.scatter(i + 0.11, np.array(all_res_sft[ckp])[:, 1], s=2, c='r', marker='x')
plt.hlines(0.5, -0.5, len(all_res) - 0.5, colors='k', linestyles='dashed', linewidth=0.5, label="0.5")
plt.xticks(np.arange(len(all_res)), all_res.keys(), rotation=45, ha="right", fontsize=8)
# plt.ylim(0, 1)
plt.ylabel("Classification score", fontsize=8)
# plt.xlabel("Pretrained model", fontsize=8)
plt.plot([], [], 'b-', label="normal")
plt.plot([], [], 'r-', label="abnormal")
plt.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=8)
# %%
# %%
from sklearn.manifold import TSNE
name = "1000genome_new_2022"
raw_dataset = load_dataset("csv",
                           data_files={"train": f"./data/{name}/train.csv",
                                       "validation": f"./data/{name}/validation.csv",
                                       "test": f"./data/{name}/test.csv"})
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model1 = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model2 = AutoModelForSequenceClassification.from_pretrained("./models/1000genome_new_2022/bert-base-uncased-sft")
emb1 = model1.get_input_embeddings()
emb2 = model2.get_input_embeddings()

# %%
import umap
s_idx = []
for i in range(100):
    s_idx.extend(tokenizer.encode(raw_dataset['test'][0]['text']))

# use UMAP to visualiza the embeddings
um = umap.UMAP(n_neighbors=15, n_components=2)
# tsne = TSNE(n_components=2)
# X_2d = tsne.fit_transform(emb1.weight[s_idx].detach().cpu().numpy())
# X_2d2 = tsne.fit_transform(emb2.weight[s_idx].detach().cpu().numpy())

X_2d = um.fit_transform(emb1.weight[s_idx].detach().cpu().numpy())
X_2d2 = um.fit_transform(emb2.weight[s_idx].detach().cpu().numpy())

plt.scatter(X_2d[:, 0], X_2d[:, 1], c='b', marker='.', label="bert-base-uncased")
plt.scatter(X_2d2[:, 0], X_2d2[:, 1], c='r', marker='.', label="bert-base-uncased-sft")

# %%


# %%
""" Print the bias of pretrained models (without SEED, run 10 times) """

pbar = tqdm(ckps, desc="ckp")
all_res_zsc = defaultdict(list)
for ckp in pbar:
    for _ in range(10):
        zsc = pipeline(task="zero-shot-classification", model=ckp)
        # check prediction with empty string
        res = zsc([" "], candidate_labels=["normal", "abnormal"])
        # print(ckp, clf(""))
        # label = int(res[0]['label'][-1:])
        # scores = [0, 0]
        # scores[label] = res[0]['score']
        # scores[1 - label] = 1 - res[0]['score']
        scores = res[0]['scores']
        all_res_zsc[ckp].append(scores)

        # if label == 0:
        #     print(f"| {ckp:<25} | {res[0]['score']:>10.4f} | {1-res[0]['score']:>10.4f} | {label:^10} |")
        # else:
        #     print(f"| {ckp:<25} | {1-res[0]['score']:>10.4f} | {res[0]['score']:>10.4f} | {label:^10} |")
        pbar.set_postfix({"ckp": ckp, "score 0": scores[0], "1": scores[1]})

# %%
""" Plot the boxplot of the biased results from pretrained models (ZSC problem) """
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(4, 3), tight_layout=True, dpi=600)
# box style
boxprops_normal = dict(linestyle='-', linewidth=1, color='b')
boxprops_abnormal = dict(linestyle='-', linewidth=1, color='r')
for i, ckp in enumerate(all_res_zsc):
    plt.boxplot(np.array(all_res_zsc[ckp])[:, 0], positions=[i - 0.11], widths=0.2, boxprops=boxprops_normal)
    plt.boxplot(np.array(all_res_zsc[ckp])[:, 1], positions=[i + 0.11], widths=0.2, boxprops=boxprops_abnormal)

plt.hlines(0.5, -0.5, len(all_res_zsc) - 0.5, colors='k', linestyles='dashed', linewidth=0.5, label="0.5")
plt.xticks(np.arange(len(all_res_zsc)), all_res_zsc.keys(), rotation=45, ha="right", fontsize=8)
# plt.ylim(0, 1)
plt.ylabel("Classification score", fontsize=8)
# plt.xlabel("Pretrained model", fontsize=8)
plt.plot([], [], 'b-', label="normal")
plt.plot([], [], 'r-', label="abnormal")
plt.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=8)
# %%
from transformers import AutoModel
# %%
# %%
