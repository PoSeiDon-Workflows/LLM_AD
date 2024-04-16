# %% [markdown]
# # Supervised fine-tuning using local data
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
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

# load local modules
sys.path.append('../')
from utils import create_dir

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.getLogger("transformers").setLevel(logging.CRITICAL)
torch.manual_seed(42)

# %% [markdown]
# ## Load data into huggingface datasets
#

# %%
name = "1000genome"
raw_dataset = load_dataset("csv",
                           data_files={"train": f"../data/{name}/train.csv",
                                       "validation": f"../data/{name}/validation.csv",
                                       "test": f"../data/{name}/test.csv"})
ckp = "bert-base-uncased"

metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])


torch.cuda.empty_cache()

# prepare the tokenizer
tokenizer = AutoTokenizer.from_pretrained(ckp)
tokenized_datasets = raw_dataset.map(lambda data: tokenizer(data['text'], truncation=True), batched=True)
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


# %%
print(raw_dataset)
print("text input:", raw_dataset['train'][0]['text'])
print("label:", raw_dataset['train'][0]['label'])

# %% [markdown]
# ## Model without supervised fine-tuning

# %%
# record time
tic = datetime.now()
res = defaultdict(dict)
# prepare the model
model = AutoModelForSequenceClassification.from_pretrained(ckp, num_labels=2).to(DEVICE)
num_params = sum(p.numel() for p in model.parameters())
# print("Number of parameters: {:,}".format(num_params))
res[ckp]["num_params"] = num_params

# evaluate the performance in test set
model.eval()
for batch in test_dataloader:
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metrics.add_batch(predictions=predictions, references=batch["labels"])

metric_res = metrics.compute()
for m in metric_res:
    res[ckp][m] = metric_res[m]
toc = datetime.now()
print("ckp:          ", ckp)
print("num param.:   ", res[ckp]['num_params'])
print("results:      ", metric_res['accuracy'])
print("Time elapsed: ", (toc - tic).total_seconds())

# %% [markdown]
# ## Model with supervised fine-tuning

# %%
tic = datetime.now()
res = defaultdict(dict)
# prepare the model
model = AutoModelForSequenceClassification.from_pretrained(ckp, num_labels=2).to(DEVICE)
num_params = sum(p.numel() for p in model.parameters())
# print("Number of parameters: {:,}".format(num_params))
res[ckp]["num_params"] = num_params

# fine-tune the model
folder = f"../models/{name}/{ckp}"
create_dir(folder)
# setup training args
training_args = TrainingArguments(output_dir=folder,
                                  overwrite_output_dir=True,
                                  save_total_limit=1,
                                  #   save_strategy="no",
                                  seed=42,
                                  auto_find_batch_size=True,
                                  num_train_epochs=3)
# train with local dataset
trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=tokenized_datasets["train"],
                  eval_dataset=tokenized_datasets["validation"],
                  data_collator=data_collator,
                  tokenizer=tokenizer,
                  )
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


try:
    trainer.train()
    # record training time
    res[ckp]["train_time"] = (datetime.now() - tic).total_seconds()

    # save to local folder
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)

    model.eval()
    predictions = trainer.predict(tokenized_datasets["test"])
    pred_labels = np.argmax(predictions.predictions, axis=1)
    metric_res = metrics.compute(predictions=pred_labels, references=predictions.label_ids)

    # print(f"ckp, {ckp}", metrics.compute())
    # metric_res = metrics.compute()
    for m in metric_res:
        res[ckp][m] = metric_res[m]
    toc = datetime.now()
    print("ckp:          ", ckp)
    print("num param.:   ", res[ckp]['num_params'])
    print("results:      ", metric_res['accuracy'])
    print("Training time:", res[ckp]["train_time"])
    print("Time elapsed: ", (toc - tic).total_seconds())
except Exception as e:
    print(ckp, e)

# %%
