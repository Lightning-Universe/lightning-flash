# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random

import networkx as nx

import flash
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.transforms import ApplyToKeys
from flash.core.utilities.imports import _TORCH_GEOMETRIC_AVAILABLE
from flash.graph.classification.data import GraphClassificationData
from flash.graph.classification.model import GraphClassifier

if _TORCH_GEOMETRIC_AVAILABLE:
    import torch_geometric
    import torch_geometric.transforms as T
    from torch_geometric.datasets import TUDataset
else:
    raise ModuleNotFoundError("Please, pip install -e '.[graph]'")

# 1. Create the DataModule
dataset = TUDataset("data", name='IMDB-BINARY').shuffle()
num_features = 136
transform = {
    "pre_tensor_transform": ApplyToKeys(DefaultDataKeys.INPUT, T.OneHotDegree(num_features - 1)),
    "to_tensor_transform": ApplyToKeys(DefaultDataKeys.INPUT, T.ToSparseTensor())
}
dm = GraphClassificationData.from_datasets(
    train_dataset=dataset[:len(dataset) // 2],
    test_dataset=dataset[len(dataset) // 2:],
    val_split=0.1,
    train_transform=transform,
    val_transform=transform,
    predict_transform=transform,
    num_features=num_features,
)
# 2. Build the task
model = GraphClassifier(num_classes=dm.num_classes)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=1)
trainer.fit(model, datamodule=dm)

# 4. Predict what's on the first 3 graphs
predictions = model.predict(dataset[:3])
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("graph_classification.pt")
