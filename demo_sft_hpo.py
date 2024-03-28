""" Hyperparameter search (HPS) for SFT with Optuna

* Task: text-classification (binary labels)
* Method: Supervised Fine-tuning
* Dataset: 1000genome

NOTE:
  * hyperparameter search for "trainer" in HF,
  * define search
"""

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BertConfig, DataCollatorWithPadding, Trainer,
                          TrainingArguments)


def ray_hp_space(trial):
    # NOTE: this should be compatible with DeepHyper
    from ray import tune
    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "per_device_train_batch_size": tune.choice([8, 16, 32, 64, 128]),
    }


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-4),
        "adam_beta1": trial.suggest_float("adam_beta1", 0.9, 0.999),
        "adam_beta2": trial.suggest_float("adam_beta2", 0.9, 0.999),
        "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-9, 1e-7),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.5, 1.0),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 10),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
    }


name = "1000genome"
ckp = "bert-base-uncased"

# load dataset
raw_dataset = load_dataset("csv",
                           data_files={"train": f"./data/{name}/train.csv",
                                       "validation": f"./data/{name}/validation.csv",
                                       "test": f"./data/{name}/test.csv"})

tokenizer = AutoTokenizer.from_pretrained(ckp)
tokenizer_datasets = raw_dataset.map(lambda data: tokenizer(data["text"], truncation=True), batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenizer_datasets = tokenizer_datasets.remove_columns(["text"])
tokenized_datasets = tokenizer_datasets.rename_column("label", "labels")
tokenized_datasets.set_format(type="torch")

# TODO: Define the variable "model_args" and "config" here
model_args = {"model_name": "sft_hps",
              "cache_dir": "./cache",
              "model_revision": "main",
              "use_auth_token": "None"}
# NOTE: add hps in Bert:
# https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertConfig
config = BertConfig()
train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['validation']
acc = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return acc.compute(predictions=predictions, references=labels)


def model_init(trial):
    return AutoModelForSequenceClassification.from_pretrained(
        ckp,
        config=config,
    )


# set hps to training arguments
training_args = TrainingArguments(
    output_dir="./models/tmp-sft",
    overwrite_output_dir=True,
    save_strategy="no",
    seed=42,
    auto_find_batch_size=True,
    num_train_epochs=3,
)

trainer = Trainer(
    model=None,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # compute_metrics=compute_metrics, # NOTE: remove metrics and it will return eval loss by default
    tokenizer=tokenizer,
    model_init=model_init,
    data_collator=data_collator,
)

best_trials = trainer.hyperparameter_search(
    direction="minimize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=20,
    # compute_objective=compute_objective,
)

print(best_trials.hyperparameters)
