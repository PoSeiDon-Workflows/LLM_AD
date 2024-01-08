# %%

# %%
import pandas as pd
import torch
from transformers import (AlbertForSequenceClassification, AutoModel,
                          AutoModelForSequenceClassification, AutoTokenizer, RobertaModel,
                          BertForSequenceClassification, pipeline)

from data_processing import build_text_data, load_tabular_data, split_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
print(DEVICE)

# %%
''' preprocess data '''
merged_df = load_tabular_data(columns=["wms_delay"])
build_text_data(df=merged_df)
df = pd.read_csv("output.csv")

# %%
ckp = "albert-base-v2"
ckp = "bert-base-uncased"
ckp = "roberta-base"
model = AutoModelForSequenceClassification.from_pretrained(ckp, num_labels=2).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(ckp)

# %%

# %%
clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
torch.cuda.empty_cache()
y_pred = []
for i in tqdm(range(len(df))):
    # tokers = tokenizer([df['text'][i]], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    # outputs = model(**tokers)
    y_pred.append(int(clf(df['text'][i])[0]["label"].split("_")[1]))
y_true = df["label"].tolist()
# inputs = tokenizer(df["text"].tolist()[:1000], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
# outputs = model(**inputs)
# outputs.logits.argmax(1)
print(classification_report(y_true, y_pred))

# %%
