# %% [markdown]
# # Dealing with Catastrophic Forgetting with Freezing
# * dataset: 1000genome, Montage
# * model: bert-base-uncased
# * model without freezing vs. model with freezing in SFT

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
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          pipeline)
import sys
sys.path.append('../')
from data_processing import build_text_data, load_tabular_data
from utils import create_dir

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
logging.getLogger("transformers").setLevel(logging.CRITICAL)

# %%
''' load data '''
name = "1000genome"
raw_dataset = load_dataset("csv",
                           data_files={"train": f"../data/{name}/train.csv",
                                       "validation": f"../data/{name}/validation.csv",
                                       "test": f"../data/{name}/test.csv"})

name2 = "montage"
raw_dataset2 = load_dataset("csv",
                            data_files={"train": f"../data/{name2}/train.csv",
                                        "validation": f"../data/{name2}/validation.csv",
                                        "test": f"../data/{name2}/test.csv"})

ckp = "bert-base-uncased"
metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained(ckp)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = raw_dataset.map(lambda data: tokenizer(data["text"], truncation=True), batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

tokenized_datasets2 = raw_dataset2.map(lambda data: tokenizer(data["text"], truncation=True), batched=True)
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets2 = tokenized_datasets2.remove_columns(["text"])
tokenized_datasets2 = tokenized_datasets2.rename_column("label", "labels")
tokenized_datasets2.set_format("torch")


# %%
''' load pretrained model '''

model = AutoModelForSequenceClassification.from_pretrained(ckp, num_labels=2).to(DEVICE)
training_args = TrainingArguments(output_dir=ckp,
                                  overwrite_output_dir=True,
                                  save_strategy="no",
                                  seed=42,
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

model.eval()
predictions = trainer.predict(tokenized_datasets["test"])
pred_labels = np.argmax(predictions.predictions, axis=1)
metric_res = metrics.compute(predictions=pred_labels, references=predictions.label_ids)
print("test metrics on pretrained model", metric_res)

# %%
model_folder = "../models/tmp_1000genome"
''' sft model on 1000genome workflows  '''
training_args = TrainingArguments(output_dir=model_folder,
                                  overwrite_output_dir=True,
                                  save_strategy="no",
                                  per_device_train_batch_size=64,
                                  per_device_eval_batch_size=64,
                                  seed=42,
                                  #   use_cpu=True,
                                  #   auto_find_batch_size=True,
                                  num_train_epochs=3)
trainer = Trainer(model=model, args=training_args,
                  train_dataset=tokenized_datasets["train"],
                  eval_dataset=tokenized_datasets["validation"],
                  data_collator=data_collator,
                  tokenizer=tokenizer,
                  #   compute_metrics=metrics
                  )
trainer.train()
model.save_pretrained(model_folder)
tokenizer.save_pretrained(model_folder)

model.eval()
predictions = trainer.predict(tokenized_datasets["test"])
pred_labels = np.argmax(predictions.predictions, axis=1)
metric_res = metrics.compute(predictions=pred_labels, references=predictions.label_ids)
print("test metrics on sft model", metric_res)

# %%
''' sft model again on montage workflows '''
old_model_folder = "../models/tmp_1000genome"
model = AutoModelForSequenceClassification.from_pretrained(old_model_folder, num_labels=2).to(DEVICE)

model_folder = "../models/tmp_1000genome_montage"
training_args = TrainingArguments(output_dir=model_folder,
                                  overwrite_output_dir=True,
                                  save_strategy="no",
                                  per_device_train_batch_size=64,
                                  per_device_eval_batch_size=64,
                                  seed=42,
                                  #   use_cpu=True,
                                  #   auto_find_batch_size=True,
                                  num_train_epochs=3)
trainer = Trainer(model=model, args=training_args,
                  train_dataset=tokenized_datasets2["train"],
                  eval_dataset=tokenized_datasets2["validation"],
                  data_collator=data_collator,
                  tokenizer=tokenizer,
                  #   compute_metrics=metrics
                  )
trainer.train()
model.save_pretrained(model_folder)
tokenizer.save_pretrained(model_folder)

model.eval()
# evaluate on 1000 genome again
predictions = trainer.predict(tokenized_datasets["test"])
pred_labels = np.argmax(predictions.predictions, axis=1)
metric_res = metrics.compute(predictions=pred_labels, references=predictions.label_ids)
print("test metrics on sft model (1000gnome, montage)", metric_res)

# %%
''' sft last layer '''
old_model_folder = "../models/tmp_1000genome"
model = AutoModelForSequenceClassification.from_pretrained(old_model_folder, num_labels=2).to(DEVICE)

# Freeze the model parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the parameters of the head of the last linear layers
for param in model.classifier.parameters():
    param.requires_grad = True

model_folder = "../models/tmp_1000genome_montage_freeze_v1"
training_args = TrainingArguments(output_dir=model_folder,
                                  overwrite_output_dir=True,
                                  save_strategy="no",
                                  per_device_train_batch_size=64,
                                  per_device_eval_batch_size=64,
                                  seed=42,
                                  #   use_cpu=True,
                                  #   auto_find_batch_size=True,
                                  num_train_epochs=1)
trainer = Trainer(model=model, args=training_args,
                  train_dataset=tokenized_datasets2["train"],
                  eval_dataset=tokenized_datasets2["validation"],
                  data_collator=data_collator,
                  tokenizer=tokenizer,
                  #   compute_metrics=metrics
                  )
trainer.train()
model.save_pretrained(model_folder)
tokenizer.save_pretrained(model_folder)

model.eval()
# evaluate on 1000 genome again
predictions = trainer.predict(tokenized_datasets["test"])
pred_labels = np.argmax(predictions.predictions, axis=1)
metric_res = metrics.compute(predictions=pred_labels, references=predictions.label_ids)
print("test metrics on sft model - freeze (1000gnome, montage)", metric_res)
# %%
