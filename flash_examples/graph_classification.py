from torch_geometric.datasets.karate import KarateClub

from flash.graph.classification.data import GraphClassificationData
from flash.graph.classification.model import GraphClassifier

dataset = KarateClub()

print(dataset[0])
