""" A demo of zero-shot classification (ZSC) on local dataset. """

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
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainingArguments, pipeline)

from data_processing import build_text_data, load_tabular_data
from utils import create_dir

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
logging.getLogger("transformers").setLevel(logging.CRITICAL)

# %%
name = "1000genome_new_2022"
raw_dataset = load_dataset("csv",
                           data_files={"train": f"./data/{name}/train.csv",
                                       "validation": f"./data/{name}/validation.csv",
                                       "test": f"./data/{name}/test.csv"})



# %%
ckps = [
    "facebook/bart-large-mnli",
    "alexandrainst/scandi-nli-large",
    "typeform/distilbert-base-uncased-mnli"
]
# "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
# "joeddav/xlm-roberta-large-xnli",
# "cross-encoder/nli-deberta-v3-small",
# "vicgalle/xlm-roberta-large-xnli-anli",
# "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",

for ckp in ckps:
    print(ckp, end=": ")
    zsc = pipeline("zero-shot-classification", model=ckp)
    res = zsc(" ", ["normal", "abnormal"])
    print(res['scores'])

# %%
