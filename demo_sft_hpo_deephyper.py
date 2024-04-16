""" Hyperparameter search (HPS) for SFT with DeeyHyper

* Task: text-classification (binary labels)
* Method: Supervised Fine-tuning
* Dataset: 1000genome
* Pre-trained model: bert-base-uncased

Ref:
* https://docs.nersc.gov/machinelearning/hpo/
* https://deephyper.readthedocs.io/en/latest/install/hpc/nersc.html
"""

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BertConfig, DataCollatorWithPadding, Trainer,
                          TrainingArguments)
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from deephyper.evaluator import Evaluator


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

train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['validation']
acc = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return acc.compute(predictions=predictions, references=labels)


def run(config):

    trainer_config = {"learning_rate": config.get("learning_rate", 5e-5),
                      "weight_decay": config.get("weight_decay", 0.0),
                      "adam_beta1": config.get("adam_beta1", 0.9),
                      "adam_beta2": config.get("adam_beta2", 0.999),
                      "adam_epsilon": config.get("adam_epsilon", 1e-8),
                      "max_grad_norm": config.get("max_grad_norm", 1.0),
                      "num_train_epochs": config.get("num_train_epochs", 3),
                      "per_device_train_batch_size": config.get("per_device_train_batch_size", 32)}

    # NOTE: not defined in problem
    model_config = {"hidden_act": config.get("hidden_act", "gelu"),
                    "hidden_dropout_prob": config.get("hidden_dropout_prob", 0.1),
                    "hidden_size": config.get("hidden_size", 768),
                    "initializer_range": config.get("initializer_range", 0.02),
                    "intermediate_size": config.get("intermediate_size", 3072),
                    "layer_norm_eps": config.get("layer_norm_eps", 1e-12),
                    "max_position_embeddings": config.get("max_position_embeddings", 512),
                    "num_attention_heads": config.get("num_attention_heads", 12),
                    "num_hidden_layers": config.get("num_hidden_layers", 12),
                    "type_vocab_size": config.get("type_vocab_size", 2),
                    "vocab_size": config.get("vocab_size", 30522)}
    # NOTE: add hps in Bert:
    # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertConfig
    bert_config = BertConfig(**model_config)

    model = AutoModelForSequenceClassification.from_pretrained(ckp, config=bert_config)

    # NOTE: add hps to training arguments
    training_args = TrainingArguments(
        output_dir="./models/tmp-sft",
        overwrite_output_dir=True,
        save_strategy="no",
        seed=42,
        auto_find_batch_size=True,
        **trainer_config
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # compute_metrics=compute_metrics, # NOTE: remove metrics and it will return eval loss by default
        tokenizer=tokenizer,
        # model_init=model,
        data_collator=data_collator,
    )
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()

    # Return the evaluation loss, maximize the -loss
    return -eval_results['eval_loss']


problem = HpProblem()
# NOTE: define hyperparameters search space
# trainer parameters
problem.add_hyperparameter((1e-6, 1e-4, "log-uniform"), "learning_rate", default_value=5e-5)
problem.add_hyperparameter((0.0, 1e-4), "weight_decay", default_value=0.0)
problem.add_hyperparameter((0.9, 0.999), "adam_beta1", default_value=0.9)
problem.add_hyperparameter((0.9, 0.999), "adam_beta2", default_value=0.999)
problem.add_hyperparameter((1e-9, 1e-7), "adam_epsilon", default_value=1e-8)
problem.add_hyperparameter((0.5, 1.0), "max_grad_norm", default_value=1.0)
problem.add_hyperparameter((3, 10), "num_train_epochs", default_value=3)
problem.add_hyperparameter([16, 32, 64, 128], "per_device_train_batch_size", default_value=32)
# pretrain config parameters

# define the evaluator to distribute the computation
# TODO: check the method compatible with NERSC for multi-gpu usage
evaluator = Evaluator.create(run,
                             method="serial",
                             method_kwargs={
                                 "num_workers": 2,
                             })

search = CBO(problem, evaluator)

results = search.search(max_evals=10)
