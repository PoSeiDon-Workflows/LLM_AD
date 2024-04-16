""" Hyperparameter search (HPS) for SFT with Optuna (with `trainer.hyperparameter_search`)

* Task: text-classification (binary labels)
* Method: Supervised Fine-tuning
* Dataset: 1000genome
* Pre-trained model: bert-base-uncased

"""

import pickle

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BertConfig, DataCollatorWithPadding, Trainer,
                          TrainingArguments)


def optuna_hp_space(trial):
    # NOTE: define the hyperparameter search space
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
ckps = [
    # "albert-base-v2",
    # "albert-large-v2",
    # "bert-base-cased",
    # "bert-base-uncased",
    "bert-large-cased",
    "bert-large-uncased",
    "distilbert-base-cased",
    "distilbert-base-uncased",
    "roberta-base",
    "roberta-large",
    "xlnet-base-cased",
    "xlnet-large-cased"
]

# load dataset
raw_dataset = load_dataset("csv",
                           data_files={"train": f"./data/{name}/train.csv",
                                       "validation": f"./data/{name}/validation.csv",
                                       "test": f"./data/{name}/test.csv"})

res = {}
for ckp in ckps:
    tokenizer = AutoTokenizer.from_pretrained(ckp)
    tokenizer_datasets = raw_dataset.map(lambda data: tokenizer(data["text"], truncation=True), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenizer_datasets = tokenizer_datasets.remove_columns(["text"])
    tokenized_datasets = tokenizer_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(type="torch")

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

        # model_init = AutoModelForSequenceClassification.from_pretrained(ckp, config=config)

    def model_init(trial):
        return AutoModelForSequenceClassification.from_pretrained(
            ckp,
            # from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            # cache_dir=model_args.cache_dir,
            # revision=model_args.model_revision,
            # token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=True,
        )

    # set hps to training arguments
    training_args = TrainingArguments(
        output_dir="./models/tmp-sft",
        overwrite_output_dir=True,
        save_strategy="no",
        seed=42,
        auto_find_batch_size=True,
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

    print(ckp, best_trials.hyperparameters)
    res[ckp] = best_trials.hyperparameters

pickle.dump(res, open("hps_optuna.pkl", "wb"))
