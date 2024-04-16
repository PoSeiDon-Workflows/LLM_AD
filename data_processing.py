""" Data processing script """
import glob
import logging
import os.path as osp

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from utils import create_dir, parse_adj

TS_FEATURES = ["ready",
               "submit",
               "execute_start",
               "execute_end",
               "post_script_start",
               "post_script_end",
               "stage_in_start",
               "stage_in_end"]
DELAY_FEATURES = ["wms_delay",
                  "queue_delay",
                  "runtime",
                  "post_script_delay",
                  "stage_in_delay",
                  "stage_out_delay"]
BYTES_FEATURES = ["stage_in_bytes",
                  "stage_out_bytes"]
KICKSTART_FEATURES = ["kickstart_executables_cpu_time"]


def load_tabular_data(name="1000genome",
                      columns=None,
                      binary=True):
    """ Load the tabular data from `raw_data` folder.

    Args:
        name (str, optional): Name of the workflow.
            Defaults to "1000genome".
        columns (list, optional): Columns of features to be select.
            Defaults to None, to select all the features.

    Returns:
        pd.DataFrame: A dataframe of combined data
    """
    data_files = {"1000genome": "1000-genome",
                  "montage": "montage",
                  "predict_future_sales": "predict-future-sales"}
    # ! the raw data is located in ../graph_nn_2/raw_data
    # TODO: replace the relative path with flowbench api
    files = glob.glob(f"./raw_data/*/{data_files[name]}*.csv")
    df_list = []
    for file in files:
        df = pd.read_csv(file, index_col=[0])
        nodes, edges = parse_adj(name)
        # change the index the same as `nodes`
        for i, node in enumerate(df.index.values):
            if node.startswith("create_dir_") or node.startswith("cleanup_"):
                new_name = node.split("-")[0]
                df.index.values[i] = new_name

        # sort node name in json matches with node in csv.
        df = df.iloc[df.index.map(nodes).argsort()]
        # df.index = df.index.map(nodes)

        # subtract the timestamp by the first timestamp (ready)
        df[TS_FEATURES] = df[TS_FEATURES].sub(df[TS_FEATURES].ready.min())

        df = df.fillna(0)
        df_list.append(df)

    # concatenate list of dataframes
    merged_df = pd.concat(df_list)

    # select features
    if columns is None:
        selected_features = DELAY_FEATURES + BYTES_FEATURES + KICKSTART_FEATURES
    else:
        if isinstance(columns, str):
            selected_features = [columns]
        else:
            selected_features = columns

    # add `label`
    if binary:
        merged_df['label'] = merged_df["anomaly_type"].map(lambda x: 0 if x == 0 else 1)
        merged_df = merged_df[selected_features + ['label']]
    else:
        # ! TODO: add multi-labels
        _multi_labels = list(merged_df["anomaly_type"].unique())
        _multi_cat = [cat.split("_")[0] for cat in _multi_labels if cat != "None"]
        label_map = {label: i + 1 for i, label in enumerate(_multi_cat) if label != "None"}
        label_map["None"] = 0
        merged_df['label'] = merged_df["anomaly_type"].map(label_map)
        merged_df = merged_df[selected_features + ['label']]

    return merged_df


def build_text_data(df,
                    folder="./",
                    name="1000genome",
                    **kwargs):
    """ Convert the tabular data into text data with columns of ['text', 'label']
        "<COLUMN> is <VALUE> <COLUMN> is <VALUE> ... ,<LABEL>"

    Args:
        df (pd.DataFrame): Dataframe of concated data.
        folder (str, optional): Folder name to be processed. Defaults to "./".
        name (str, optional): Name of the workflow. Defaults to "1000genome".

    Returns:
        str: File name of the output csv file.
    """
    output_dir = osp.join(folder, name)
    create_dir(output_dir)
    outfile = osp.join(output_dir, "all.csv")
    with open(outfile, "w") as f:
        f.write("text,label\n")
        for index, row in df.iterrows():
            row_str = ""
            for col in df.columns:
                if col != "label":
                    # row_str += f"{col} is {row[col]} "
                    row_str += f"{' '.join(col.split('_'))} is {row[col]} "
            row_str += f",{int(row['label'])}"
            row_str += "\n"
            f.write(row_str)

    return outfile


def split_dataset(name="1000genome",
                  pretrained_model="distilbert-base-uncased"):
    """Split dataset into train/val/test with ratio of 0.8/0.1/0.1.

    Args:
        name (str, optional): Name of the workflow.
            Defaults to "1000genome".
        pretrained_model (str, optional): Name of pretrained model.
            Defaults to "distilbert-base-uncased".

    Returns:
        DatasetDict: A dictionary of train/val/test dataset.
    """
    logging.warning("This function is deprecated, use `load_dataset` instead.")
    # tokenize step
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    def tokenize_function(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True)

    abs_path = build_text_data(name)
    ds = Dataset.from_csv(abs_path)
    ds_dict = DatasetDict()
    train_test_ds = ds.train_test_split(test_size=0.2, shuffle=True)
    test_val_ds = train_test_ds["test"].train_test_split(test_size=0.5, shuffle=True)
    ds_dict["train"] = train_test_ds["train"]
    ds_dict["val"] = test_val_ds["train"]
    ds_dict["test"] = test_val_ds["test"]
    ds_dict_encoded = ds_dict.map(tokenize_function, batched=True, batch_size=None)

    return ds_dict_encoded


if __name__ == "__main__":
    wns = ["1000genome", "montage", "predict_future_sales"]
    data_folder = "./data_v2"
    for name in wns:
        logging.info(f"processing {name}")

        df = load_tabular_data(name=name, columns=DELAY_FEATURES)
        print(df.describe())
        fn = build_text_data(df=df, folder=data_folder, name=name)
        df = pd.read_csv(fn)

        # shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # split into train/validation/test
        train_ratio, validation_ratio, test_raio = 0.8, 0.1, 0.1
        total_size = len(df)
        train_df = df[: int(total_size * train_ratio)]
        validation_df = df[int(total_size * train_ratio): int(total_size * (train_ratio + validation_ratio))]
        test_df = df[int(total_size * (train_ratio + validation_ratio)):]
        train_df = pd.DataFrame(train_df)
        validation_df = pd.DataFrame(validation_df)
        test_df = pd.DataFrame(test_df)

        # save to local files
        # logging.info(f"save to {data_folder}/{name}")
        # train_df.to_csv(f"{data_folder}/{name}/train.csv", index=False)
        # validation_df.to_csv(f"{data_folder}/{name}/validation.csv", index=False)
        # test_df.to_csv(f"{data_folder}/{name}/test.csv", index=False)
