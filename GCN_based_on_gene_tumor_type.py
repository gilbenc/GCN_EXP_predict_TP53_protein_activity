# Check GCN for other tumor types, genes

import traceback
import warnings
import pandas as pd
import numpy as np
import copy
import sklearn
import torch

from models.mlp import MLP
from models.gcn import GCN
from data.datasets import TCGADataset
from data.gene_graphs import GeneManiaGraph, RegNetGraph, HumanNetV2Graph, \
    FunCoupGraph

PATH = "/home/user/gil/Expression project/GCN_EXP_predict_TP53_protein_activity/data/gene_tumor_specific"

### pick gene and tumor_type for model run.
genes = ["TP53", "KRAS"]
tumor_types = ["LUAD", "COAD", "PAAD"]
# gene = "TP53"
# tumor_type = "BRCA"

seed = 0
cuda = torch.cuda.is_available()

graph_dict = {"regnet": RegNetGraph, "genemania": GeneManiaGraph,
              "humannetv2": HumanNetV2Graph, "funcoup": FunCoupGraph}
# # Select graph and set variables
# if args.graph:
#     # Check graph arg is valid
#     assert args.graph in graph_dict.keys()
#     graph_name = args.graph
#     gene_graph = graph_dict[graph_name]()
#     is_first_degree = True
# else:
#     is_first_degree = False
#     graph_name = "all_nodes"
# # adj for GCN
# adj = gene_graph.adj()

# Read in data: TCGA
TCGA_dataset = TCGADataset()

# tuning
num_layer = 2
channels = 30
embedding = 40
dropout = False
batch_size = 10
is_first_degree = True

# load graph and create adj matrix.
graph = "funcoup"
gene_graph = graph_dict[graph]()
# adj for GCN
adj = gene_graph.adj()

results = []
best_auc = 0
best_acc = 0
best_auc_model = None
best_acc_model = None
fixed_params = {
    "model": "GCN",
    "graph": graph,
    # "gene": gene,
    # "tumor_type": tumor_type
    "seed": seed,
    # "train_size": train_size,
    "num_layer": num_layer,
    "channels": channels,
    "embedding": embedding,
    "batch": batch_size,
    "dropout": dropout,
}

# graph analysis
for gene in genes:
    for tumor_type in tumor_types:
        dataset = copy.deepcopy(TCGA_dataset)
        # samples and labels modified based on gene and tumor_type selection.
        # TCGA BRCA samples only
        nb_examples = pd.read_csv(
            PATH + tumor_type + "_samples.csv",
            # relative python path to subdirectory
            header=0,  # first row is header.
            index_col=0
        )
        dataset.df = dataset.df.reindex(nb_examples.T.iloc[0])
        dataset.sample_names = nb_examples.T.iloc[0].tolist()
        # labels:
        labels = pd.read_csv(
            PATH + tumor_type + "_" + gene + "_labels.csv",
            # relative python path to subdirectory
            header=0,  # first row is header.
            index_col=0
        )
        labels = labels.T.iloc[0]
        dataset.labels = labels.values

        train_size = round(0.75 * len(nb_examples))
        test_size = len(nb_examples) - train_size

        # model = MLP(column_names=dataset.df.columns, num_layer=1, dropout=False,
        #             train_valid_split=0.5, cuda=cuda, metric=sklearn.metrics.roc_auc_score,
        #             channels=16, batch_size=10, lr=0.0007, weight_decay=0.00000001,
        #             verbose=False, patience=5, num_epochs=10, seed=seed,
        #             full_data_cuda=True, evaluate_train=False)

        # monitor number of epochs by is_first_degree (worked well for genemania)
        # if is_first_degree:
        #     num_epochs = 30
        # else:
        #     num_epochs = 10

        model = GCN(column_names=dataset.df.columns, name="GCN_lay3_chan32_emb32_dropout", cuda=True,
                    num_layer=num_layer, channels=channels, embedding=embedding,  batch_size=batch_size,
                    dropout=dropout)


        experiment = {
            "gene": gene,
            "tumor_type": tumor_type,
            "model": "GCN",
            # "graph": graph,
            # "num_layer": num_layer,
            # "channels": channels,
            "train_size": train_size,
            # "embedding": embedding,
            # "dropout": dropout,
            # "batch": batch_size
        }

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset.df, dataset.labels,
                                                                                    stratify=dataset.labels,
                                                                                    train_size=train_size,
                                                                                    test_size=test_size,
                                                                                    random_state=seed)

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

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train, adj)
                model.eval()
                with torch.no_grad():
                    y_hat = model.predict(X_test)
            auc = sklearn.metrics.roc_auc_score(y_test, np.argmax(y_hat, axis=1))
            acc = sklearn.metrics.accuracy_score(y_test, np.argmax(y_hat, axis=1))
            print("auc:", auc, " acc: ", acc)
            experiment["auc"] = auc
            experiment["acc"] = acc
            results.append(experiment)
            if auc > best_auc:
                best_auc = copy.deepcopy(auc)
                best_auc_model = copy.deepcopy(model)
            if acc > best_acc:
                best_acc = copy.deepcopy(acc)
                best_acc_model = copy.deepcopy(model)
            model.best_model = None  # cleanup
            del model
            torch.cuda.empty_cache()
        except Exception:
            tb = traceback.format_exc()
            experiment['error'] = tb
            print(tb)
print(fixed_params)
print(*results, sep="\n")
print("best AUC: ", best_auc, " best ACC: ", best_acc)

