import random

import networkx as nx

import flash
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.transforms import ApplyToKeys
from flash.core.utilities.imports import _GRAPH_AVAILABLE
from flash.graph.classification.data import GraphClassificationData
from flash.graph.embedding.model import GraphEmbedder

if _GRAPH_AVAILABLE:
    import torch_geometric.transforms as T
    from torch_geometric.datasets import TUDataset
else:
    raise ModuleNotFoundError("Please, pip install -e '.[graph]'")

# 1. Create the DataModule
dataset = TUDataset(root="data", name="KKI").shuffle()

# 2. Load a previously trained model (for example, if you have previously run a classification task)
model = GraphEmbedder.load_from_checkpoint("graph_classification.pt")

# 3. Predict what's on the first 3 graphs
predictions = model.predict(dataset[:3])
print(predictions)
