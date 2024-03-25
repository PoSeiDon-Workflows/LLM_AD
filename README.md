# Unlocking the Potential of Large Language Models for Anomaly Detection in Scientific Workflows: A Supervised Fine-Tuning Approach

## Data processing

Run scripts in `data_processing` folder and create folders under `data` folder.
Each folder contains the data for one dataset, and four csv files: `all.csv`, `train.csv`, `validation.csv`, `test.csv` for all data, training data, validation data and test data respectively.

The folder structure and the md5sum of the files are as follows:
```bash
├── 1000genome_new_2022
│   ├── all.csv         2cda70f14707683102426ae945623ab2
│   ├── test.csv        e4f808adaa5aa110bf1db26dea66658e
│   ├── train.csv       12c6554dab3594071afd5af6b159479e
│   └── validation.csv  fbff302e70a365447683a7ee557d9c47
├── montage
│   ├── all.csv         b83b1ec871d4c50acd384f821f06d43a
│   ├── test.csv        5a1da93d60cc703c0d12c673cdb0adc9
│   ├── train.csv       0374689cae2a7dcc6139043f702fc2f9
│   └── validation.csv  5b2acfe8b9e459e521c02b0c578a3030
└── predict_future_sales
    ├── all.csv         7638fc62503946a0e4db22a40eded33c
    ├── test.csv        6c04b27a497b1e7f56e4a23d6ae556c0
    ├── train.csv       c27a9039bc27cdaecafa1e41d64ffb41
    └── validation.csv  d0bedeaebbe8b7f765fe30aa0fef8654
```

## Diagram of the proposed approach

* Pipeline of the proposed approach
![image](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/full_nlp_pipeline-dark.svg)

* Transformer and head
![image](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/transformer_and_head-dark.svg)

## Using transformers for anomaly detection

### text-classification task

* load dataset

```python
from datasets import load_dataset
name = "1000genome_new_2022"
raw_dataset = load_dataset("csv",
                           data_files={"train": f"./data/{name}/train.csv",
                                       "validation": f"./data/{name}/validation.csv",
                                       "test": f"./data/{name}/test.csv"})
```

* load model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
```

### zero-shot-classification task


### new comparisons
<!-- 1000 genome training epoch: 3 -->
  | model                   | train time | accuracy | f1     | precision | recall |
  | ----------------------- | ---------: | -------- | ------ | --------- | ------ |
  | albert-base-v2          |   897.8760 | 0.8027   | 0.6936 | 0.7080    | 0.6797 |
  | bert-base-cased         |   844.2449 | 0.8137   | 0.7182 | 0.7137    | 0.7228 |
  | bert-base-uncased       |   804.1232 | 0.8143   | 0.7212 | 0.7116    | 0.7310 |
  | distilbert-base-cased   |   439.5890 | 0.8099   | 0.7109 | 0.7105    | 0.7114 |
  | distilbert-base-uncased |   420.6979 | 0.8139   | 0.7070 | 0.7322    | 0.6835 |
  | roberta-base            |   764.7687 | 0.7802   | 0.6323 | 0.7019    | 0.5753 |
  | logbert [1][1]          |  3443.0343 | 0.8024   | 0.7800 | 0.7019    | 0.7589 |
  | deeplog [2][2]          |  4531.0869 | 0.8099   | 0.4341 | 0.9671    | 0.2798 |
  | loganomaly [3][3]       |  6885.2573 | 0.8139   | 0.5868 | 0.9671    | 0.4239 |


* logbert:
  * LogBERT is a model for log anomaly detection that leverages the BERT (Bidirectional Encoder Representations from Transformers) model, which is a transformer-based machine learning technique for natural language processing. LogBERT treats log anomaly detection as a natural language processing task, considering the log key sequence as a sentence and the log key as a word.
  * The model is trained to understand the normal patterns of these "sentences" and "words". When it encounters a log key sequence that deviates from the normal patterns it has learned, it identifies it as an anomaly.
* deeplog:
  * DeepLog uses Long Short-Term Memory (LSTM) networks to model a system log as a natural language sequence. It learns the patterns of 'normal' log sequences and can then detect anomalies as deviations from these learned patterns.
* loganomaly:
  * Sequential anomaly detection: It uses an LSTM-based model to learn the normal patterns of log sequences. If a new log sequence deviates from these learned patterns, it is flagged as a sequential anomaly.
  * Quantitative anomaly detection: It calculates the log key count within a time window and uses a Gaussian distribution to model the normal log key count. If the actual log key count deviates from the normal count, it is flagged as a quantitative anomaly.
  * 
[1]: Guo, Haixuan, Shuhan Yuan, and Xintao Wu. "Logbert: Log anomaly detection via bert." 2021 international joint conference on neural networks (IJCNN). IEEE, 2021.

[2]: Du, Min, et al. "Deeplog: Anomaly detection and diagnosis from system logs through deep learning." Proceedings of the 2017 ACM SIGSAC conference on computer and communications security. 2017.

[3]: Meng, Weibin, et al. "Loganomaly: Unsupervised detection of sequential and quantitative anomalies in unstructured logs." IJCAI. Vol. 19. No. 7. 2019.
