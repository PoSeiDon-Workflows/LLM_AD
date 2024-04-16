# %% [markdown]
# # Transfer learning with SFT
#
# * Follow the [README.md](../README.md) to set up the environment and data sources.
# * Dataset: `1000 Genome` (binary labels), `Montage` (binary labels)
# * Pretrained model: `bert-base-uncased`
# * SFT model on local dataset, and predict on the other dataset with zero-shot learning.

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
from sklearn import pipeline
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)
import sys
sys.path.append("../")

from data_processing import build_text_data, load_tabular_data
from utils import create_dir

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.getLogger("transformers").setLevel(logging.CRITICAL)

# %%
name1 = "1000genome"
name2 = "montage"

ckp = "bert-base-uncased"

ckp1 = f"../models/{name1}/{ckp}-sft"
ckp2 = f"../models/{name2}/{ckp}-sft"

raw_dataset1 = load_dataset("csv",
                            data_files={"train": f"../data/{name1}/train.csv",
                                        "validation": f"../data/{name1}/validation.csv",
                                        "test": f"../data/{name1}/test.csv"})
raw_dataset2 = load_dataset("csv",
                            data_files={"train": f"../data/{name2}/train.csv",
                                        "validation": f"../data/{name2}/validation.csv",
                                        "test": f"../data/{name2}/test.csv"})

metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

# %%
''' evaluate the model on test set of dataset 2 from pretrained model of dataset 1 '''

tokenizer1 = AutoTokenizer.from_pretrained(ckp1)
tokenized_datasets2 = raw_dataset2.map(lambda data: tokenizer1(data["text"], truncation=True), batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer1)

tokenized_datasets2 = tokenized_datasets2.remove_columns(["text"])
tokenized_datasets2 = tokenized_datasets2.rename_column("label", "labels")
tokenized_datasets2.set_format("torch")

model1 = AutoModelForSequenceClassification.from_pretrained(ckp1, num_labels=2)
training_args = TrainingArguments(output_dir=ckp1,
                                  overwrite_output_dir=True,
                                  save_strategy="no",
                                  per_device_train_batch_size=64,
                                  per_device_eval_batch_size=64,
                                  seed=42,
                                  use_cpu=True,
                                  # auto_find_batch_size=True,
                                  num_train_epochs=3)
trainer = Trainer(model=model1, args=training_args,
                  train_dataset=tokenized_datasets2["train"],
                  eval_dataset=tokenized_datasets2["validation"],
                  data_collator=data_collator,
                  tokenizer=tokenizer1,
                  #   compute_metrics=metrics
                  )
model1.eval()
predictions2 = trainer.predict(tokenized_datasets2["test"])
pred_labels = np.argmax(predictions2.predictions, axis=1)
metric_res = metrics.compute(predictions=pred_labels, references=predictions2.label_ids)
print("from model 1 to dataset 2", metric_res)

#  %%
''' evaluate the model on test set of dataset 1 from pretrained model of dataset 2 '''

tokenizer2 = AutoTokenizer.from_pretrained(ckp2)
tokenized_datasets1 = raw_dataset2.map(lambda data: tokenizer2(data["text"], truncation=True), batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer2)

tokenized_datasets1 = tokenized_datasets1.remove_columns(["text"])
tokenized_datasets1 = tokenized_datasets1.rename_column("label", "labels")
tokenized_datasets1.set_format("torch")

model2 = AutoModelForSequenceClassification.from_pretrained(ckp2, num_labels=2)
training_args = TrainingArguments(output_dir=ckp2,
                                  overwrite_output_dir=True,
                                  save_strategy="no",
                                  per_device_train_batch_size=64,
                                  per_device_eval_batch_size=64,
                                  seed=42,
                                  use_cpu=True,
                                  # auto_find_batch_size=True,
                                  num_train_epochs=3)
trainer = Trainer(model=model2, args=training_args,
                  train_dataset=tokenized_datasets1["train"],
                  eval_dataset=tokenized_datasets1["validation"],
                  data_collator=data_collator,
                  tokenizer=tokenizer2,
                  #   compute_metrics=metrics
                  )
model2.eval()
predictions1 = trainer.predict(tokenized_datasets1["test"])
pred_labels = np.argmax(predictions1.predictions, axis=1)
metric_res = metrics.compute(predictions=pred_labels, references=predictions1.label_ids)
print("from model 2 to dataset 1", metric_res)
