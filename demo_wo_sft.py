import logging
import pickle
from collections import defaultdict

import evaluate
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.getLogger("transformers").setLevel(logging.CRITICAL)

name = "1000genome_new_2022"
raw_dataset = load_dataset("csv",
                           data_files={"train": f"./data/{name}/train.csv",
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

res = defaultdict(dict)
for ckp in ckps:
    # clear GPU cache
    torch.cuda.empty_cache()

    # prepare the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckp)
    tokenized_datasets = raw_dataset.map(lambda data: tokenizer(data["text"], truncation=True), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=64, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=64, collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        tokenized_datasets["test"], batch_size=64, collate_fn=data_collator
    )

    # prepare the model
    model = AutoModelForSequenceClassification.from_pretrained(ckp, num_labels=2).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    # print("Number of parameters: {:,}".format(num_params))
    res[ckp]["num_params"] = num_params

    model.eval()
    for batch in test_dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metrics.add_batch(predictions=predictions, references=batch["labels"])

    # print(f"ckp, {ckp}", metrics.compute())
    metric_res = metrics.compute()
    for m in metric_res:
        res[ckp][m] = metric_res[m]
    # res[ckp] = metric_res
    print(f"ckp, {ckp}", metric_res)

for ckp in res:
    print(f"model: {ckp}", end="  ")
    for m in res[ckp]:
        print(f"{m}: {res[ckp][m]:.4f}", end="  ")
    print()
pickle.dump(res, open("res_w_sft.pkl", "wb"))
