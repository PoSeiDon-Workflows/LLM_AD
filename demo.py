# %%
%load_ext autoreload
%autoreload 2

# %%
import pandas as pd
from transformers import AutoTokenizer, BertForSequenceClassification
from data_processing import build_text_data, load_tabular_data, split_dataset
from transformers import pipeline
import torch
# %%

# merged_df = load_dataframe()
# path = build_vocab(df=merged_df)
# ds_dict_encoded = split_dataset(pretrained_model="bert")

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
ckp = "bert-base-uncased"
model = BertForSequenceClassification.from_pretrained(ckp, num_labels=2).to(DEVICE)
# %%
tokenizer = AutoTokenizer.from_pretrained(ckp)

# %%
df = pd.read_csv("output.csv")
df.iloc[0]

# %%

# clf = pipeline("text-classification", model=model)
clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
# %%
# for text in df["text"].tolist()[:10]:
#     print(clf(text.to(DEVICE)))
# %%
inputs = tokenizer(df["text"].tolist()[:200], padding=True, truncation=True, return_tensors="pt").to(DEVICE)

outputs = model(**inputs)
# %%
from transformers import AutoModelForSequenceClassification

# TODO: build a benchmark for sequenceclassification without SFT
