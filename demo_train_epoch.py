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

name = "1000genome_new_2022"
raw_dataset = load_dataset("csv",
                           data_files={"train": f"./data/{name}/train.csv",
                                       "validation": f"./data/{name}/validation.csv",
                                       "test": f"./data/{name}/test.csv"})

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

res = defaultdict(dict)
ckp = "bert-base-uncased"
test_metrics = []
train_time = []

# prepare the tokenizer
tokenizer = AutoTokenizer.from_pretrained(ckp)
tokenized_datasets = raw_dataset.map(lambda data: tokenizer(data["text"], truncation=True), batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# prepare the model
model = AutoModelForSequenceClassification.from_pretrained(ckp, num_labels=2).to(DEVICE)

for epoch in range(50):
    # clear GPU cache
    torch.cuda.empty_cache()
    tic = datetime.now()

    # fine-tune the model
    folder = f"./models/tmp-sft"
    create_dir(folder)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(folder, num_labels=2).to(DEVICE)
    except BaseException:
        model = AutoModelForSequenceClassification.from_pretrained(ckp, num_labels=2).to(DEVICE)

    # training_args = TrainingArguments(output_dir=folder,
    #                                   overwrite_output_dir=True,
    #                                   save_strategy="no",
    #                                   per_device_train_batch_size=64,
    #                                   per_device_eval_batch_size=64,
    #                                   seed=42,
    #                                   #   use_cpu=True,
    #                                   # auto_find_batch_size=True,
    #                                   num_train_epochs=3)
    # # NOTE: special case for xlnet-large-cased
    # if ckp == "xlnet-large-cased":
    #     training_args = TrainingArguments(output_dir=folder,
    #                                       overwrite_output_dir=True,
    #                                       save_strategy="no",
    #                                       seed=42,
    #                                       #   per_device_train_batch_size=64,
    #                                       #   per_device_eval_batch_size=64,
    #                                       #   use_cpu=True,
    #                                       auto_find_batch_size=True,
    #                                       num_train_epochs=3)

    training_args = TrainingArguments(output_dir=folder,
                                      overwrite_output_dir=True,
                                      save_strategy="no",
                                      seed=42,
                                      per_device_train_batch_size=64,
                                      per_device_eval_batch_size=64,
                                      #   use_cpu=True,
                                      #   auto_find_batch_size=True,
                                      num_train_epochs=1)
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=tokenized_datasets["train"],
                      eval_dataset=tokenized_datasets["validation"],
                      data_collator=data_collator,
                      tokenizer=tokenizer,
                      #   compute_metrics=metrics
                      )
    try:
        trainer.train()
        # record training time
        res[ckp]["train_time"] = (datetime.now() - tic).total_seconds()
        train_time.append(res[ckp]["train_time"])


        model.save_pretrained(folder)
        tokenizer.save_pretrained(folder)

        model.eval()
        predictions = trainer.predict(tokenized_datasets["test"])
        pred_labels = np.argmax(predictions.predictions, axis=1)
        metric_res = metrics.compute(predictions=pred_labels, references=predictions.label_ids)
        test_metrics.append(metric_res)

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

pickle.dump({"test_metrics": test_metrics, "train_time": train_time}, open("train_epoch_res.pkl", "wb"))

# for ckp in res:
#     print(f"model: {ckp}", end="  ")
#     for m in res[ckp]:
#         print(f"{m}: {res[ckp][m]:.4f}", end="  ")
#     print()
# # pickle.dump(res, open("res_sft.pkl", "wb"))
