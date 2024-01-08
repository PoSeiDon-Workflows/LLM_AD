# %% [markdown]
# # LLM for System Log Anomaly Detection
# ## Step 1: prepare data


# %%
import glob

import pandas as pd
from sklearn.metrics import accuracy_score

from utils import parse_adj

# %% []
''' demo with a single file '''
files = glob.glob("../graph_nn_2/data_new/*/1000-genome*.csv")

# # single file
# df = pd.read_csv(files[0], index_col=[0])
# nodes, edges = parse_adj("1000genome_new_2022")
# # change the index the same as `nodes`
# for i, node in enumerate(df.index.values):
#     if node.startswith("create_dir_") or node.startswith("cleanup_"):
#         new_name = node.split("-")[0]
#         df.index.values[i] = new_name

# # sort node name in json matches with node in csv.
# # df = df.iloc[df.index.map(nodes).argsort()]
# df.index = df.index.map(nodes)

# ts_features = [
#     'ready',
#     'submit',
#     'execute_start',
#     'execute_end',
#     'post_script_start',
#     'post_script_end',
#     'stage_in_start',
#     'stage_in_end']

# df[ts_features] = df[ts_features].sub(df[ts_features].ready.min())
# df.fillna(0)
# # ts_features = ['ready', 'submit', 'execute_start', 'execute_end', 'post_script_start', 'post_script_end']
# delay_features = ["wms_delay", "queue_delay", "runtime",
#                   "post_script_delay", "stage_in_delay", "stage_out_delay"]
# bytes_features = ["stage_in_bytes", "stage_out_bytes"]
# kickstart_features = ["kickstart_executables_cpu_time"]

# selected_features = delay_features + bytes_features + kickstart_features
# df = df[selected_features + ['anomaly_type']]
# df.fillna(0)

# df.to_csv("tmp.csv")
# %%

''' process all files '''
df_list = []

for file in files:
    df = pd.read_csv(files[0], index_col=[0])
    nodes, edges = parse_adj("1000genome_new_2022")
    # change the index the same as `nodes`
    for i, node in enumerate(df.index.values):
        if node.startswith("create_dir_") or node.startswith("cleanup_"):
            new_name = node.split("-")[0]
            df.index.values[i] = new_name

    # sort node name in json matches with node in csv.
    # df = df.iloc[df.index.map(nodes).argsort()]
    df.index = df.index.map(nodes)

    ts_features = [
        'ready',
        'submit',
        'execute_start',
        'execute_end',
        'post_script_start',
        'post_script_end',
        'stage_in_start',
        'stage_in_end']

    df[ts_features] = df[ts_features].sub(df[ts_features].ready.min())
    df.fillna(0)
    # ts_features = ['ready', 'submit', 'execute_start', 'execute_end', 'post_script_start', 'post_script_end']
    delay_features = ["wms_delay", "queue_delay", "runtime",
                      "post_script_delay", "stage_in_delay", "stage_out_delay"]
    bytes_features = ["stage_in_bytes", "stage_out_bytes"]
    kickstart_features = ["kickstart_executables_cpu_time"]

    selected_features = delay_features + bytes_features + kickstart_features
    # df['label'] = df["anomaly_type"].map({"None": 0, "cpu_2": 1})
    df['label'] = df["anomaly_type"].map(lambda x: 1 if x != "None" else 0)
    df = df[selected_features + ['label']]
    df.fillna(0)
    df_list.append(df)

merged_df = pd.concat(df_list)
# %%
''' build the vocab as "<column> is <value> and <column> is <value> and ... ,<anomaly_type> \n" '''

# ! DEBUG: text is incorrect.
with open('output.csv', 'w') as f:
    f.write("text,label\n")
    for index, row in merged_df.iterrows():
        row_str = ''
        for col in merged_df.columns:
            if col != "label":
                row_str += f'{col} is {row[col]} '
        row_str += f",{int(row['label'])}"
        row_str += '\n'
        f.write(row_str)


# %%

''' tokenize the data '''
from transformers import AutoTokenizer, BertTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


# %%

''' build the dataset '''
from datasets import Dataset, DatasetDict

psd = Dataset.from_csv("output.csv")
psd_dict = DatasetDict()
train_test_dataset = psd.train_test_split(train_size=0.8, shuffle=True)
psd_dict['train'] = train_test_dataset['train']
test_val_dataset = train_test_dataset['test'].train_test_split(train_size=0.5, shuffle=True)
psd_dict['validation'] = test_val_dataset['train']
psd_dict['test'] = test_val_dataset['test']
psd_dict_encoded = psd_dict.map(tokenize_function, batched=True, batch_size=None)
# psd_dict_encoded

# %%
''' init the model '''
import torch
from transformers import AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 2
model = (AutoModelForSequenceClassification
         .from_pretrained(checkpoint, num_labels=num_labels)
         .to(device))
# model


# %%

from transformers import Trainer, TrainingArguments

batch_size = 8
logging_steps = len(psd_dict_encoded["train"]) // batch_size
logging_steps = 10000
model_name = f"{checkpoint}-finetuned-psd_tiny"
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=2,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  logging_steps=2000,
                                  log_level="error",
                                  optim='adamw_torch'
                                  )

# training_args

# %%


torch.cuda.empty_cache()


def get_accuracy(preds):
    from sklearn.metrics import accuracy_score
    predictions = preds.predictions.argmax(axis=-1)
    labels = preds.label_ids
    accuracy = accuracy_score(preds.label_ids, preds.predictions.argmax(axis=-1))
    return {'accuracy': accuracy}


trainer = Trainer(model=model,
                  compute_metrics=get_accuracy,
                  args=training_args,
                  train_dataset=psd_dict_encoded["train"],
                  eval_dataset=psd_dict_encoded["validation"],
                  tokenizer=tokenizer)
trainer.train()
# %%

''' testing '''
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

pred = trainer.predict(psd_dict_encoded['test'])

prob = softmax(pred.predictions, 1)
test_acc = accuracy_score(pred.label_ids, pred.predictions.argmax(axis=-1))
print(f"test accuracy  {test_acc:.4f}")
conf = [prob[i, label] - prob[i, 1 - label] for i, label in enumerate(pred.label_ids)]
plt.scatter(np.arange(len(conf)), conf)
plt.ylabel("confidence")
plt.show()

# %%

''' load model '''

from transformers import pipeline

model_name = "distilbert-base-cased-finetuned-psd_tiny"
classifier = pipeline('text-classification', model=model_name)
input_text = psd_dict['test'][0]['text']
classifier(input_text)
# %%

''' prediction with test data '''
print("Input: ", psd_dict['test'][0]['text'])
print("Prediction: ", classifier(psd_dict['test'][0]['text']))
print("Ground truth: ", psd_dict['test'][0]['label'])

# %%
splits = psd_dict['test'][2]['text'].split("and")
for i in range(len(splits)):
    print(" and ".join(splits[:i + 1]))
    print("prediction", classifier(" and ".join(splits[:i + 1])))

# %%

''' prediction with montage data '''

files_2 = glob.glob("../graph_nn_2/data_new/*/montage*.csv")

# df = pd.read_csv(files_2[0], index_col=[0])
# nodes, edges = parse_adj("montage")

# for i, node in enumerate(df.index.values):
#     if node.startswith("create_dir_") or node.startswith("cleanup_"):
#         new_name = node.split("-")[0]
#         df.index.values[i] = new_name

# # sort node name in json matches with node in csv.
# # df = df.iloc[df.index.map(nodes).argsort()]
# df.index = df.index.map(nodes)

# ts_features = [
#     'ready',
#     'submit',
#     'execute_start',
#     'execute_end',
#     'post_script_start',
#     'post_script_end',
#     'stage_in_start',
#     'stage_in_end']

# df[ts_features] = df[ts_features].sub(df[ts_features].ready.min())
# df.fillna(0)
# # ts_features = ['ready', 'submit', 'execute_start', 'execute_end', 'post_script_start', 'post_script_end']
# delay_features = ["wms_delay", "queue_delay", "runtime",
#                   "post_script_delay", "stage_in_delay", "stage_out_delay"]
# bytes_features = ["stage_in_bytes", "stage_out_bytes"]
# kickstart_features = ["kickstart_executables_cpu_time"]

# selected_features = delay_features + bytes_features + kickstart_features
# df = df[selected_features + ['anomaly_type']]
# df.fillna(0)


# process all files
df_list = []

for file in files:
    df = pd.read_csv(files[0], index_col=[0])
    nodes, edges = parse_adj("montage")
    # change the index the same as `nodes`
    for i, node in enumerate(df.index.values):
        if node.startswith("create_dir_") or node.startswith("cleanup_"):
            new_name = node.split("-")[0]
            df.index.values[i] = new_name

    # sort node name in json matches with node in csv.
    # df = df.iloc[df.index.map(nodes).argsort()]
    df.index = df.index.map(nodes)

    ts_features = [
        'ready',
        'submit',
        'execute_start',
        'execute_end',
        'post_script_start',
        'post_script_end',
        'stage_in_start',
        'stage_in_end']

    df[ts_features] = df[ts_features].sub(df[ts_features].ready.min())
    df.fillna(0)
    # ts_features = ['ready', 'submit', 'execute_start', 'execute_end', 'post_script_start', 'post_script_end']
    delay_features = ["wms_delay", "queue_delay", "runtime",
                      "post_script_delay", "stage_in_delay", "stage_out_delay"]
    bytes_features = ["stage_in_bytes", "stage_out_bytes"]
    kickstart_features = ["kickstart_executables_cpu_time"]

    selected_features = delay_features + bytes_features + kickstart_features
    df['label'] = df["anomaly_type"].map({"None": 0, "cpu_2": 1})
    df = df[selected_features + ['label']]
    df.fillna(0)
    df_list.append(df)

merged_df = pd.concat(df_list)

with open('output_montage.csv', 'w') as f:
    f.write("text,label\n")
    for index, row in merged_df.iterrows():
        row_str = ''
        for col in df.columns:
            if col != "label":
                row_str += f'{col} is {row[col]} and '
        row_str += f",{int(row['label'])}"
        row_str += '\n'
        f.write(row_str)


psd_montage = Dataset.from_csv("output_montage.csv")
psd_dict_montage = DatasetDict()
train_test_dataset_montage = psd_montage.train_test_split(train_size=0.8, shuffle=True)
test_val_dataset_montage = train_test_dataset_montage['test'].train_test_split(train_size=0.5, shuffle=True)
psd_dict_montage['train'] = train_test_dataset_montage['train']
psd_dict_montage['validation'] = test_val_dataset_montage['train']
psd_dict_montage['test'] = test_val_dataset_montage['test']
psd_dict_encoded_montage = psd_dict_montage.map(tokenize_function, batched=True, batch_size=None)

pred = trainer.predict(psd_dict_encoded_montage['test'])

prob = softmax(pred.predictions, 1)

# ! too good to be true
accuracy_score(pred.label_ids, pred.predictions.argmax(axis=-1))

# %%
''' Prediction with test in montage data '''
print("Input: ", psd_dict_montage['test'][0]['text'])
print("Prediction: ", classifier(psd_dict_montage['test'][0]['text']))
print("Ground truth: ", psd_dict_montage['test'][0]['label'])

# %%
''' prediction with NERSC data '''
tt = "nid00752 on data_00064 starts 2020-0921T18:37Z09.706 and spotfind_start takes 2 seconds and stopped due to not enough spots 3"
print("Input: ", tt)
print("Prediction: ", classifier(
    "nid00752 on data_00064 starts 2020-0921T18:37Z09.706 and spotfind_start takes 2 seconds and stopped due to not enough spots 3"))


# %%
# ! plot the results
import matplotlib.pyplot as plt


def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha='center')


fig = plt.figure(figsize=(4, 3), tight_layout=True)
text = psd_dict_montage['test'][0]['text']
plt.text(0.5, 1.5, text, fontsize=8, ha='center', va='top', wrap=True)
plt.bar([0, 1], [0.1, 0.9], width=0.5)
plt.xticks([0, 1], ["normal", "abnormal"])
plt.ylim(0, 1.5)
addlabels([0, 1], [0.1, 0.9])
# %%


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3), tight_layout=True)
ax1.axis('off')
ax1.text(0.5, 1.5, text, fontsize=8, ha='center', va='top', wrap=True)

ax2.bar([0, 1], [0.1, 0.9], width=0.5)
# %%
splits = psd_dict_montage['test'][0]['text'].split("and")
for i in range(len(splits)):
    print(" and ".join(splits[:i + 1]))
    print("prediction", classifier(" and ".join(splits[:i + 1])))
    label = classifier(" and ".join(splits[:i + 1]))[0]['label'].split("_")[1]
    score = classifier(" and ".join(splits[:i + 1]))[0]['score']
    if label == '1':
        y = [1 - score, score]
    else:
        y = [score, 1 - score]
    fig = plt.figure(figsize=(4, 3), tight_layout=True)
    plt.bar([0, 1], y, width=0.5)
    plt.savefig(f"res_{i:02d}.png")

import matplotlib.pyplot as plt
# %%
# plot the results
import numpy as np

res = np.array([[0, 0.9961193799972534],
                [0, 0.998943030834198],
                [0, 0.6797028183937073],
                [0, 0.9984676241874695],
                [1, 0.5547395944595337],
                [1, 0.9983280301094055],
                [0, 0.9990252256393433],
                [1, 0.9996693134307861],
                [1, 0.9996728897094727]])

# %%
for i in range(len(res)):

    label = res[i][0]
    score = res[i][1]
    if label == 0:
        y = [1 - score, score]
    else:
        y = [score, 1 - score]
    fig = plt.figure(figsize=(4, 3), tight_layout=True)
    plt.bar([0, 1], y, width=0.5)
    plt.ylim(0, 1)
    # plt.text(0.5, 1.5, " ".join(splits[:i+1]), fontsize=8, ha='center', va='top', wrap=True)
    plt.savefig(f"res_{i:02d}.png")

# %%
