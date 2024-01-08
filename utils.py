""" Utility functions in data processing.

License: TBD
"""
import argparse
import functools
import glob
import json
import os
import os.path as osp
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.simplefilter("ignore", category=UserWarning)


def process_args():
    """ Process args of inputs

    Returns:
        dict: Parsed arguments.
    """
    workflows = ["1000genome",
                 "nowcast-clustering-8",
                 "nowcast-clustering-16",
                 "wind-clustering-casa",
                 "wind-noclustering-casa",
                 "1000genome_new_2022",
                 "montage",
                 "predict_future_sales",
                 "all"
                 ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", "-w",
                        type=str,
                        default="1000genome",
                        help="Name of workflow.",
                        choices=workflows)
    parser.add_argument("--binary",
                        action="store_true",
                        help="Toggle binary classification.")
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="GPU id. `-1` for CPU only.")
    parser.add_argument("--epoch",
                        type=int,
                        default=500,
                        help="Number of epoch in training.")
    parser.add_argument("--hidden_size",
                        type=int,
                        default=64,
                        help="Hidden channel size.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="Batch size.")
    parser.add_argument("--conv_blocks",
                        type=int,
                        default=2,
                        help="Number of convolutional blocks")
    parser.add_argument("--train_size",
                        type=float,
                        default=0.6,
                        help="Train size [0.5, 1). And equal split on validation and testing.")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay for Adam.")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
                        help="Dropout in neural networks.")
    parser.add_argument("--feature_option",
                        type=str,
                        default="v1",
                        help="Feature option.")
    parser.add_argument("--seed",
                        type=int,
                        default=-1,
                        help="Fix the random seed. `-1` for no random seed.")
    parser.add_argument("--path", "-p",
                        type=str,
                        default=".",
                        help="Specify the root path of file.")
    parser.add_argument("--log",
                        action="store_true",
                        help="Toggle to log the training")
    parser.add_argument("--logdir",
                        type=str,
                        default="runs",
                        help="Specify the log directory.")
    parser.add_argument("--force",
                        action="store_true",
                        help="To force reprocess datasets.")
    parser.add_argument("--balance",
                        action="store_true",
                        help="Enforce the weighted loss function.")
    parser.add_argument("--verbose", "-v",
                        action="store_true",
                        help="Toggle for verbose output.")
    parser.add_argument("--output", "-o",
                        action="store_true",
                        help="Toggle for pickle output file.")
    parser.add_argument("--anomaly_cat",
                        type=str,
                        default="all",
                        help="Specify the anomaly set.")
    parser.add_argument("--anomaly_level",
                        nargs="*",
                        help="Specify the anomaly levels. Multiple inputs.")
    parser.add_argument("--anomaly_num",
                        type=str,
                        help="Specify the anomaly num from nodes.")
    args = vars(parser.parse_args())

    return args


def parse_adj(workflow):
    """ Processing adjacency file.

    Args:
        workflow (str): Workflow name.

    Raises:
        NotImplementedError: No need to process the workflow `all`.

    Returns:
        tuple: (dict, list)
            dict: Dictionary of nodes.
            list: List of directed edges.
    """
    adj_folder = osp.join(osp.dirname(osp.abspath(__file__)), "adjacency_list_dags")
    if workflow == "all":
        raise NotImplementedError
    else:
        adj_file = osp.join(adj_folder, f"{workflow.replace('-', '_')}.json")
    adj = json.load(open(adj_file))

    if workflow == "predict_future_sales":
        nodes = {}
        for idx, node_name in enumerate(adj.keys()):
            nodes[node_name] = idx

        edges = []
        for u in adj:
            for v in adj[u]:
                edges.append((nodes[u], nodes[v]))
    else:
        # build dict of node: {node_name: idx}
        nodes = {}
        for idx, node_name in enumerate(adj.keys()):
            if node_name.startswith("create_dir_") or node_name.startswith("cleanup_"):
                node_name = node_name.split("-")[0]
                nodes[node_name] = idx
            else:
                nodes[node_name] = idx

        # build list of edges: [(target, source)]
        edges = []
        for u in adj:
            for v in adj[u]:
                if u.startswith("create_dir_") or u.startswith("cleanup_"):
                    u = u.split("-")[0]
                if v.startswith("create_dir_") or v.startswith("cleanup_"):
                    v = v.split("-")[0]
                edges.append((nodes[u], nodes[v]))

    return nodes, edges


def print_dataset_info(dataset):
    """ Print the dataset information.

    Args:
        dataset (PyG.dataset): Dataset object.
    """
    print(dataset)
    print(f"dataset                 {dataset.name} \n",
          f"# of graphs             {len(dataset)} \n",
          f"# of graph labels       {dataset.num_classes} \n",
          f"# of node features      {dataset.data.num_node_features} \n",
          f"# of nodes per graph    {dataset[0].num_nodes} \n",
          f"# of edges per graph    {dataset[0].num_edges} \n",
          "##" * 20 + "\n"
          )


def create_dir(path):
    """ Create a dir where the processed data will be stored

    Args:
        path (str): Path to create the folder.
    """
    dir_exists = os.path.exists(path)

    if not dir_exists:
        try:
            os.makedirs(path)
            print("The {} directory is created.".format(path))
        except Exception as e:
            print("Error: {}".format(e))
            exit(-1)


def process_data(graphs, drop_columns):
    """ Process the columns for graphs. """
    raise NotImplementedError


def save_ckpt(filename, model, results_train, results_test, cg_dict=None, **kwargs):
    """ Save a pre-trained pytorch model to checkpoint.

    Args:
        filename (str): Filename of saved checkpoint
        model (class instance): Model instance.
        results_train (dict): Results of training.
        results_test (dict): Results of testing.
        cg_dict (dict): A dictionary of sampled computation graphs.
    """
    import torch
    torch.save({
        "epoch": kwargs.num_epochs,
        "model_type": kwargs.explainer_name,
        "optimizer": kwargs.optimizer,
        "results_train": results_train,
        "results_test": results_test,
        "model_state": model.state_dict(),
        "cg": cg_dict
    }, filename)


def load_ckpt(filename, device, **kwargs):
    """ Load a pre-trained pytorch model from checkpoint.

    Args:
        filename (str): Filename to save checkpoint.
        device (str): CUDA or CPU.
    """
    import torch
    print("loading model")
    print(filename)
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        ckpt = torch.load(filename, map_location=device)
    else:
        print("Checkpoint does not exist!")
        print("Checked path -- {}".format(filename))
        print("Make sure you have provided the correct path!")
        print("You may have forgotten to train a model for this dataset.")
        print()
        print("To train one of the paper's models, run the following")
        print(">> python train.py --dataset=DATASET_NAME")
        print()
        raise Exception("File not found.")
    return ckpt


def deprecated(func):
    """ This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.

    Args:
        func (function): Function name.

    Returns:
        object: return from `func`.
    """
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


@deprecated
def parse_data(flag, json_path, classes):
    """ Parse the json file into graphs.

    Args:
        flag (str): Flag name.
        json_path (str): Json file path.
        classes (_type_): _description_

    Returns:
        dict: Graph with keys: y, edge_index, x
    """
    counter = 0
    edge_index = []
    lookup = {}
    graphs = []
    # columns = ['type', 'ready',
    #            'submit', 'execute_start', 'execute_end', 'post_script_start',
    #            'post_script_end', 'wms_delay', 'pre_script_delay', 'queue_delay',
    #            'runtime', 'post_script_delay', 'stage_in_delay', 'stage_out_delay']

    # REVIEW:
    # runtime = execute_end - execute_start
    # post_script_delay = post_script_end - post_script_start
    columns = ['type',
               'ready',
               'submit',
               #    'execute_start',
               #    'execute_end',
               #    'post_script_start',
               #    'post_script_end',
               'wms_delay',
               'pre_script_delay',
               'queue_delay',
               'runtime',
               'post_script_delay',
               'stage_in_delay',
               'stage_out_delay']

    # columns = ['type',
    #            'is_clustered',
    #            'runtime',
    #            'post_script_delay',
    #            'pre_script_delay',
    #            'queue_delay',
    #            'stage_in_delay',
    #            'stage_out_delay',
    #            'wms_delay',
    #            'stage_in_bytes',
    #            'stage_out_bytes',
    #            'kickstart_executables_cpu_time',
    #            'kickstart_status',
    #            'kickstart_executables_exitcode'
    #            ]
    # columns = ['type', 'ready', 'submit', 'wms_delay', 'pre_script_delay', 'queue_delay',
    #            'runtime', 'post_script_delay', 'stage_in_delay', 'stage_out_delay']
    with open(json_path, "r") as f:
        adjacency_list = json.load(f)

        for node in adjacency_list:
            lookup[node] = counter
            counter += 1

        for node in adjacency_list:
            for e in adjacency_list[node]:
                edge_index.append([lookup[node], lookup[e]])
        for d in os.listdir("data"):
            for f in glob.glob(os.path.join("data", d, flag + "*")):
                try:
                    if d.split("_")[0] in classes:
                        graph = {"y": classes[d.split("_")[0]],
                                 "edge_index": edge_index,
                                 "x": []}
                        features = pd.read_csv(f, index_col=[0])
                        features = features.fillna(0)
                        # features = features.replace('', -1, regex=True)

                        for node in lookup:
                            if node.startswith("create_dir_") or node.startswith("cleanup_"):
                                new_l = node.split("-")[0]
                            else:
                                new_l = node
                            job_features = features[features.index.str.startswith(new_l)][columns].values.tolist()[0]

                            if len(features[features.index.str.startswith(new_l)]) < 1:
                                continue
                            if job_features[0] == 'auxiliary':
                                job_features[0] = 0
                            if job_features[0] == 'compute':
                                job_features[0] = 1
                            if job_features[0] == 'transfer':
                                job_features[0] = 2
                            # REVIEW: what's the line below
                            job_features = [-1 if x != x else x for x in job_features]
                            graph['x'].insert(lookup[node], job_features)

                        t_list = []
                        for i in range(len(graph['x'])):
                            t_list.append(graph['x'][i][1])
                        minim = min(t_list)

                        for i in range(len(graph['x'])):
                            lim = graph['x'][i][1:7]
                            lim = [v - minim for v in lim]
                            graph['x'][i][1:7] = lim
                            graphs.append(graph)
                except BaseException:
                    print("Error with the file's {} format.".format(f))
    return graphs


def norm_feature(X, fill_nan=0.0):
    """ Standard rescale the features to [0, 1].

    .. math::
        x = (x-x.min()) / (x.max() - x.min())

    Args:
        X (np.ndarray): Feature matrix with dim (W, N, F).

    Returns:
        np.ndarray: Normalized matrix.
    """
    # min/max over W and N and normalize F.
    v_min = X.min(axis=(0, 1))
    v_max = X.max(axis=(0, 1))
    X_norm = (X - v_min) / (v_max - v_min)
    np.nan_to_num(X_norm, False, nan=fill_nan)
    return X_norm


def eval_metrics(y_true, y_pred, metric=None, average="weighted", **kwargs):
    """Evaluate the models

    Args:
        y_true (np.array): True y labels.
        y_pred (np.array): Predicted y labels.
        metric (str, optional): Option of ['acc', 'f1', 'prec', 'roc_auc', 'conf_mat'].
                                Defaults to None, which eval all metrics
        average (str, optional): This parameter is required for multiclass/multilabel targets.
                                Defaults to "weighted".

    Returns:
        dict or float: metric results
    """
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score)

    if metric is None:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average=average)
        prec = precision_score(y_true, y_pred, average=average)
        recall = recall_score(y_true, y_pred, average=average)
        roc_auc = roc_auc_score(y_true, y_pred, average=average)
        conf_mat = confusion_matrix(y_true, y_pred)
        return {"acc": acc,
                "f1": f1,
                "prec": prec,
                "recall": recall,
                "roc_auc": roc_auc,
                "conf_mat": conf_mat}
    else:
        if metric == 'acc':
            res = accuracy_score(y_true, y_pred)
        elif metric == "f1":
            res = f1_score(y_true, y_pred, average=average)
        elif metric == "prec":
            res = precision_score(y_true, y_pred, average=average)
        elif metric == "recall":
            res = recall_score(y_true, y_pred, average=average)
        elif metric == "roc_auc":
            res = roc_auc_score(y_true, y_pred, average=average)
        elif metric == "conf_mat":
            res = confusion_matrix(y_true, y_pred)
        return res


def split_train_val_test(n, train_size=0.6, val_size=0.2, test_size=0.2, **kwargs):
    """ Split the n samples into train, validation and testing.

    Args:
        n (int): Number of samples
        train_size (float, optional): Proportion of train size. Defaults to 0.6.
        val_size (float, optional): Proportion of validation size. Defaults to 0.2.
        test_size (float, optional): Proportion of testing size. Defaults to 0.2.

    Returns:
        tuple: Tuple of np.array representing the indices of train/val/testing.
    """
    assert train_size + val_size + test_size == 1
    idx = np.arange(n)
    random_state = np.random.RandomState(seed=0) if "random_state" not in kwargs else kwargs.get("random_state")
    train_idx, test_idx = train_test_split(idx, train_size=0.6, random_state=random_state)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=random_state)
    return train_idx, val_idx, test_idx


def init_model(args):
    """ Init model for PyGOD

    Args:
        args (dict): Args from argparser.

    Returns:
        object: Model object.
    """
    from random import choice

    from pygod.models import (ANOMALOUS, CONAD, DOMINANT, DONE, GAAN, GCNAE,
                              GUIDE, MLPAE, SCAN, AdONE, AnomalyDAE, Radar)
    from pyod.models.lof import LOF
    from sklearn.ensemble import IsolationForest
    if not isinstance(args, dict):
        args = vars(args)
    dropout = [0, 0.1, 0.3]
    lr = [0.1, 0.05, 0.01]
    weight_decay = 0.01

    if args['dataset'] == 'inj_flickr':
        # sampling and minibatch training on large dataset flickr
        batch_size = 64
        num_neigh = 3
        epoch = 2
    else:
        batch_size = 0
        num_neigh = -1
        epoch = 300

    model_name = args['model']
    gpu = args['gpu']

    # if hasattr(args, 'epoch'):
    epoch = args.get('epoch', 200)

    if args['dataset'] == 'reddit':
        # for the low feature dimension dataset
        hid_dim = [32, 48, 64]
    else:
        hid_dim = [32, 64, 128, 256]

    if args['dataset'][:3] == 'inj':
        # auto balancing on injected dataset
        alpha = [None]
    else:
        alpha = [0.8, 0.5, 0.2]

    if model_name == "adone":
        return AdONE(hid_dim=choice(hid_dim),
                     weight_decay=weight_decay,
                     dropout=choice(dropout),
                     lr=choice(lr),
                     epoch=epoch,
                     gpu=gpu,
                     batch_size=batch_size,
                     num_neigh=num_neigh)
    elif model_name == 'anomalydae':
        hd = choice(hid_dim)
        return AnomalyDAE(embed_dim=hd,
                          out_dim=hd,
                          weight_decay=weight_decay,
                          dropout=choice(dropout),
                          theta=choice([10., 40., 90.]),
                          eta=choice([3., 5., 8.]),
                          lr=choice(lr),
                          epoch=epoch,
                          gpu=gpu,
                          alpha=choice(alpha),
                          batch_size=batch_size,
                          num_neigh=num_neigh)
    elif model_name == 'conad':
        return CONAD(hid_dim=choice(hid_dim),
                     weight_decay=weight_decay,
                     dropout=choice(dropout),
                     lr=choice(lr),
                     epoch=epoch,
                     gpu=gpu,
                     alpha=choice(alpha),
                     batch_size=batch_size,
                     num_neigh=num_neigh)
    elif model_name == 'dominant':
        return DOMINANT(hid_dim=choice(hid_dim),
                        weight_decay=weight_decay,
                        dropout=choice(dropout),
                        lr=choice(lr),
                        epoch=epoch,
                        gpu=gpu,
                        alpha=choice(alpha),
                        batch_size=batch_size,
                        num_neigh=num_neigh)
    elif model_name == 'done':
        return DONE(hid_dim=choice(hid_dim),
                    weight_decay=weight_decay,
                    dropout=choice(dropout),
                    lr=choice(lr),
                    epoch=epoch,
                    gpu=gpu,
                    batch_size=batch_size,
                    num_neigh=num_neigh)
    elif model_name == 'gaan':
        return GAAN(noise_dim=choice([8, 16, 32]),
                    hid_dim=choice(hid_dim),
                    weight_decay=weight_decay,
                    dropout=choice(dropout),
                    lr=choice(lr),
                    epoch=epoch,
                    gpu=gpu,
                    alpha=choice(alpha),
                    batch_size=batch_size,
                    num_neigh=num_neigh)
    elif model_name == 'gcnae':
        return GCNAE(hid_dim=choice(hid_dim),
                     weight_decay=weight_decay,
                     dropout=choice(dropout),
                     lr=choice(lr),
                     epoch=epoch,
                     gpu=gpu,
                     batch_size=batch_size,
                     num_neigh=num_neigh)
    elif model_name == 'guide':
        return GUIDE(a_hid=choice(hid_dim),
                     s_hid=choice([4, 5, 6]),
                     weight_decay=weight_decay,
                     dropout=choice(dropout),
                     lr=choice(lr),
                     epoch=epoch,
                     gpu=gpu,
                     alpha=choice(alpha),
                     batch_size=batch_size,
                     num_neigh=num_neigh,
                     cache_dir='./tmp')
    elif model_name == "mlpae":
        return MLPAE(hid_dim=choice(hid_dim),
                     weight_decay=weight_decay,
                     dropout=choice(dropout),
                     lr=choice(lr),
                     epoch=epoch,
                     gpu=gpu,
                     batch_size=batch_size)
    elif model_name == 'lof':
        return LOF()
    elif model_name == 'if':
        return IsolationForest()
    elif model_name == 'radar':
        return Radar(lr=choice(lr), gpu=gpu)
    elif model_name == 'anomalous':
        return ANOMALOUS(lr=choice(lr), gpu=gpu)
    elif model_name == 'scan':
        return SCAN(eps=choice([0.3, 0.5, 0.8]), mu=choice([2, 5, 10]))


def wasserstein_distance(C, p, q, backend="cvxpy"):
    r"""Computes the Wasserstein distance between two probability distributions.

    .. math::
        W(p, q) = \inf_{\gamma \in \Pi(p, q)} \sum_{i, j} \gamma_{i, j} |i - j|

    Args:
        C (torch.tensor): Cost matrix with dim (m, n).
        p (torch.tensor): Probability distribution with dim (m, ).
        q (torch.tensor): Probability distribution with dim (n, ).
        backend (str, optional): Backend to use. Defaults to "cvxpy".

    Returns:
        torch.tensor: Wasserstein distance.
    """
    if backend == "cvxpy":
        import cvxpy as cp

        X = cp.Variable((len(p), len(q)))
        objective = cp.Minimize(cp.trace(X, C))
        # x = cp.Variable(len(p))
        # objective = cp.Minimize(cp.sum(cp.multiply(p, x)))
        constraints = [X.sum(0) == p, X.sum(1) == q, X >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        return prob.value

    elif backend == "torch":
        # TODO: implement the torch version using cvxpylayers
        # example here: https://github.com/cvxgrp/cvxpylayers#pytorch
        import torch
        raise NotImplementedError
