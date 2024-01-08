# %%
import logging
import pickle
from collections import defaultdict
from datetime import datetime

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from data_processing import build_text_data, load_tabular_data
from utils import create_dir

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
logging.getLogger("transformers").setLevel(logging.CRITICAL)

# %%
name = "1000genome_new_2022"
data_folder = "data_v2"
raw_dataset = load_dataset("csv",
                           data_files={"train": f"./{data_folder}/{name}/train.csv",
                                       "validation": f"./{data_folder}/{name}/validation.csv",
                                       "test": f"./{data_folder}/{name}/test.csv"})

ckps = [
    # "albert-base-v2",
    # "albert-large-v2",
    # "bert-base-cased",
    "bert-base-uncased",
    # "bert-large-cased",
    # "bert-large-uncased",
    # "distilbert-base-cased",
    # "distilbert-base-uncased",
    # "roberta-base",
    # "roberta-large",
    # "xlnet-base-cased",
    # "xlnet-large-cased"
]

metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

# %%
res = defaultdict(dict)
for ckp in ckps[:]:
    # clear GPU cache
    torch.cuda.empty_cache()
    folder = f"./models/{name}/{ckp}-sft"

    # prepare the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(folder)
    tokenized_datasets = raw_dataset.map(lambda data: tokenizer(data["text"], truncation=True), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    # prepare the model
    model = AutoModelForSequenceClassification.from_pretrained(folder, num_labels=2).to(DEVICE)
    res[folder]["num_params"] = model.num_parameters()

    # fine-tune the model
    # ckp = f"./models/{name}/{ckp}-sft"
    # create_dir(ckp)
    training_args = TrainingArguments(output_dir=folder,
                                      overwrite_output_dir=True,
                                      save_strategy="no",
                                      per_device_train_batch_size=64,
                                      per_device_eval_batch_size=64,
                                      seed=42,
                                      #   use_cpu=True,
                                      # auto_find_batch_size=True,
                                      num_train_epochs=3)
    # NOTE: special case for xlnet-large-cased
    if ckp == "xlnet-large-cased":
        training_args = TrainingArguments(output_dir=folder,
                                          overwrite_output_dir=True,
                                          save_strategy="no",
                                          seed=42,
                                          #   per_device_train_batch_size=64,
                                          #   per_device_eval_batch_size=64,
                                          #   use_cpu=True,
                                          auto_find_batch_size=True,
                                          num_train_epochs=3)
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=tokenized_datasets["train"],
                      eval_dataset=tokenized_datasets["validation"],
                      data_collator=data_collator,
                      tokenizer=tokenizer,
                      #   compute_metrics=metrics
                      )
    try:
        # trainer.train()
        # record training time
        # res[ckp]["train_time"] = (datetime.now() - tic).total_seconds()

        # model.save_pretrained(folder)
        # tokenizer.save_pretrained(folder)

        model.eval()
        predictions = trainer.predict(tokenized_datasets["test"])
        pred_labels = np.argmax(predictions.predictions, axis=1)
        metric_res = metrics.compute(predictions=pred_labels, references=predictions.label_ids)
        # for batch in test_dataloader:
        #     batch = {k: v.to(DEVICE) for k, v in batch.items()}
        #     with torch.no_grad():
        #         outputs = model(**batch)
        #     logits = outputs.logits
        #     predictions = torch.argmax(logits, dim=-1)
        #     metrics.add_batch(predictions=predictions, references=batch["labels"])

        # print(f"ckp, {ckp}", metrics.compute())
        # metric_res = metrics.compute()
        for m in metric_res:
            res[ckp][m] = metric_res[m]
        # res[ckp] = metric_res
        print(ckp, metric_res)
    except Exception as e:
        print(ckp, e)
    finally:
        continue


# %%
for ckp in res:
    print(f"model: {ckp}", end="  ")
    for m in res[ckp]:
        print(f"{m}: {res[ckp][m]:.4f}", end="  ")
    print()

# %%
from scipy.special import softmax

n_test = len(predictions.label_ids)
true_labels = predictions.label_ids
pred_labels = np.argmax(predictions.predictions, axis=-1)
pred_prob = softmax(predictions.predictions, axis=-1)
conf = []
for i in range(n_test):
    # if pred_labels[i] == predictions.label_ids[i]:
    conf.append(pred_prob[i, predictions.label_ids[i]] - pred_prob[i, 1 - predictions.label_ids[i]])
conf = np.array(conf)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(4, 4), dpi=600, tight_layout=True)
for i in range(n_test):
    if true_labels[i] == 0 and conf[i] > 0:
        plt.scatter(i, conf[i], marker='.', c='g', alpha=0.5)
    elif true_labels[i] == 0 and conf[i] < 0:
        plt.scatter(i, conf[i], marker='.', c='r', alpha=0.5)
    elif true_labels[i] == 1 and conf[i] > 0:
        plt.scatter(i, conf[i], marker='v', c='b', alpha=0.5)
    elif true_labels[i] == 1 and conf[i] < 0:
        plt.scatter(i, conf[i], marker='v', c='k', alpha=0.5)


import re

# %%
# pipeline to evaluate the text dynamically
from transformers import pipeline

clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=DEVICE)

# ! debug:
dyn_pos = []
for i in range(n_test // 10):
    substr = re.findall(r'\w+\s+is\s+(?:\d+\.\d+|na)', raw_dataset["test"][i]["text"])

    for j in range(len(substr)):
        if int(clf(substr[:j + 1])[0]['label'][-1]) == true_labels[i]:
            dyn_pos.append(j)
            break
