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
import torch

import flash
from flash.core.utilities.imports import example_requires
from flash.graph import GraphClassificationData, GraphEmbedder

example_requires("graph")

from torch_geometric.datasets import TUDataset  # noqa: E402

# 1. Create the DataModule
dataset = TUDataset(root="data", name="KKI")
datamodule = GraphClassificationData.from_datasets(
    predict_dataset=dataset[:3],
    batch_size=4,
)

# 2. Load a previously trained GraphClassifier
model = GraphEmbedder.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/0.6.0/graph_classification_model.pt")

# 3. Generate embeddings for the first 3 graphs
trainer = flash.Trainer(gpus=torch.cuda.device_count())
predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)
