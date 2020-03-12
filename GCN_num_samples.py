
# Comparison of performance for reduced number of samples.

import pickle
import argparse
import traceback
import warnings
import pandas as pd
import numpy as np

import itertools
import sklearn
import torch

from models.mlp import MLP
from models.gcn import GCN
from data.datasets import TCGADataset
from data.gene_graphs import GeneManiaGraph, RegNetGraph, HumanNetV2Graph, \
    FunCoupGraph
from data.utils import record_result
from tqdm import tqdm

PATH = "/home/user/gil/Expression project/GCN_EXP_predict_TP53_protein_activity/data/gene_tumor_specific"

seed = 0
cuda = torch.cuda.is_available()

# Graphs
graph_dict = {"regnet": RegNetGraph, "genemania": GeneManiaGraph,
              "humannetv2": HumanNetV2Graph, "funcoup": FunCoupGraph}
# GCN
# model_name = "GCN"
# graph = "humannetv2"
# # create gene graph
# gene_graph = graph_dict[graph]()
# # adj for GCN
# adj = gene_graph.adj()
# is_first_degree = True

# MLP
model_name = "MLP"
graph = "all nodes"
is_first_degree = False


# Dataset: TCGA
dataset = TCGADataset()
# load TCGA BRCA samples
nb_examples = pd.read_csv(
    PATH + "BRCA_samples.csv",  # relative python path to subdirectory
    header=0,
    index_col=0
)
dataset.df = dataset.df.loc[nb_examples.T.iloc[0]]
dataset.sample_names = nb_examples['x'].tolist()
# labels: load TCGA_BRCA
labels = pd.read_csv(
    PATH + "BRCA_labels.csv",  # relative python path to subdirectory
    header=0,  # first row is header.
    index_col=0
)
labels = labels.T.iloc[0]
dataset.labels = labels.values

# tuning
gene = "TP53"
channels = 40
dropout = True
embedding = 50
num_layer = 3
batch_size = 10

results = []

fixed_params = {
    "gene": gene,
    "graph": graph,
    "seed": seed,
    "num_layer": num_layer,
    "channels": channels,
    "embedding": embedding,
    "batch": batch_size,
    "dropout": dropout,
}


# grid search
# for num_layer in [2, 3, 4]:
#     for channels in [30, 40, 50]:
#         for embedding in [30, 40, 50]:
#             for dropout in [True, False]:
#                 for batch_size in [5, 8, 12, 20]:

test_size = 209

# template for running with changing train_size
for remove_size in [0, 100, 200, 300, 400, 500, 600, 700, 800]:
    if remove_size == 0:
        test_size = 109
        train_size = 1100
        X_keep = dataset.df
        y_keep = dataset.labels
    else:
        train_size = 1000 - remove_size
        X_keep, X_remove, y_keep, y_remove = sklearn.model_selection.train_test_split(dataset.df, dataset.labels,
                                                                                      stratify=dataset.labels,
                                                                                      train_size=(train_size+test_size),
                                                                                      test_size=remove_size,
                                                                                      random_state=seed)

    model = MLP(column_names=dataset.df.columns, num_layer=1, dropout=False,
                train_valid_split=0.5, cuda=cuda, metric=sklearn.metrics.roc_auc_score,
                channels=16, batch_size=10, lr=0.0007, weight_decay=0.00000001,
                verbose=False, patience=5, num_epochs=10, seed=seed,
                full_data_cuda=True, evaluate_train=False)

    # monitor number of epochs by is_first_degree (worked well for genemania)
    # if is_first_degree:
    #     num_epochs = 30
    # else:
    #     num_epochs = 10
    # model = GCN(column_names=dataset.df.columns, name="GCN_lay3_chan40_emb50_dropout", cuda=True,
    #             num_layer=num_layer, channels=channels, embedding=embedding,  batch_size=batch_size,
    #             dropout=dropout)

    experiment = {
        # "gene": gene,
        "model": model_name,
        "graph": graph,
        "train_size": train_size,
    }

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_keep, y_keep,
                                                                                stratify=y_keep,
                                                                                train_size=train_size,
                                                                                test_size=test_size,
                                                                                random_state=seed)
    print(len(X_train), ", ", len(y_train), ", ", len(X_test), ", ", len(y_test))
    # Training

    if is_first_degree:
        neighbors = list(gene_graph.first_degree(gene)[0])
        neighbors = [n for n in neighbors if n in X_train.columns.values]
        X_train = X_train.loc[:, neighbors].copy()
        X_test = X_test.loc[:, neighbors].copy()
    else:
        X_train = X_train.copy()
        X_test = X_test.copy()

    try:
        # Don't include expression of enquired gene?
        # X_train[gene] = 1
        # X_test[gene] = 1
        if model_name == "GCN":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # run GCN
                model.fit(X_train, y_train, adj)
                # run MLP
                model.eval()
                with torch.no_grad():
                    y_hat = model.predict(X_test)
        if model_name == "MLP":
            model.fit(X_train, y_train)
            y_hat = model.predict(X_test)
        auc = sklearn.metrics.roc_auc_score(y_test, np.argmax(y_hat, axis=1))
        acc = sklearn.metrics.accuracy_score(y_test, np.argmax(y_hat, axis=1))
        print("auc:", auc, " acc: ", acc)
        experiment["auc"] = auc
        experiment["acc"] = acc
        results.append(experiment)
        model.best_model = None  # cleanup
        del model
        torch.cuda.empty_cache()
    except Exception:
        tb = traceback.format_exc()
        experiment['error'] = tb
        print(tb)
print(fixed_params)
print(*results, sep="\n")




