""" load from local pretrained, and evaluate on the test set """

# %%
%load_ext autoreload
%autoreload 2

# %%
import logging
import pickle
from collections import defaultdict

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sympy import per
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, pipeline)

from data_processing import build_text_data, load_tabular_data
from utils import create_dir

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.getLogger("transformers").setLevel(logging.CRITICAL)


# %%
name = "1000genome_new_2022"
# df = load_tabular_data(name=name)
# fn = build_text_data(df=df, folder="./data", name=name)
# df = pd.read_csv(fn)
# shuffle the dataset
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# split into train/validation/test
# train_ratio, validation_ratio, test_raio = 0.8, 0.1, 0.1
# total_size = len(df)
# train_df = df[: int(total_size * train_ratio)]
# validation_df = df[int(total_size * train_ratio): int(total_size * (train_ratio + validation_ratio))]
# test_df = df[int(total_size * (train_ratio + validation_ratio)):]
# train_df = pd.DataFrame(train_df)
# validation_df = pd.DataFrame(validation_df)
# test_df = pd.DataFrame(test_df)
# save to local files
# train_df.to_csv(f"./data/train_{name}.csv", index=False)
# validation_df.to_csv(f"./data/validation_{name}.csv", index=False)
# test_df.to_csv(f"./data/test_{name}.csv", index=False)

# %%
raw_dataset = load_dataset("csv",
                           data_files={
                               "train": f"./data/{name}/train.csv",
                               "validation": f"./data/{name}/validation.csv",
                               "test": f"./data/{name}/test.csv"})

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

for ckp in ckps:
    torch.cuda.empty_cache()

    # prepare the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckp)
    tokenized_datasets = raw_dataset.map(lambda data: tokenizer(data["text"], truncation=True), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    # load the model
    model = AutoModelForSequenceClassification.from_pretrained(ckp, num_labels=2).to(DEVICE)

    # sft model
    sft_dir = f"./models/{ckp}-sft"
    create_dir(sft_dir)
    training_args = TrainingArguments(output_dir=sft_dir,
                                      overwrite_output_dir=True,
                                      save_strategy="no",
                                      per_device_train_batch_size=64,
                                      per_device_eval_batch_size=64,
                                      #   use_cpu=True,
                                      num_train_epochs=1)
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=tokenized_datasets["train"],
                      eval_dataset=tokenized_datasets["validation"],
                      data_collator=data_collator,
                      tokenizer=tokenizer,
                      )
    predictions = trainer.predict(tokenized_datasets["test"])
    metric_res = metrics.compute(predictions=predictions.predictions.argmax(1), references=predictions.label_ids)

    print(ckp, metric_res)
