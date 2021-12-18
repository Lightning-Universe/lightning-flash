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
from flash.graph import GraphClassificationData, GraphNodeClassifier

example_requires("graph")

from torch_geometric.datasets import Planetoid  # noqa: E402

# 1. Create the DataModule
dataset = Planetoid(root="data", name="Cora")
dataset.validation_mask = dataset.val_mask

# The dataset is the same and should contain the masks. Or they can be different datasets without masks.
datamodule = GraphClassificationData.from_datasets(
    train_dataset=dataset,  # the dataset should contain a dataset.train_mask,
    val_dataset=dataset,  # the dataset should contain a dataset.validation_mask,
)

# 2. Build the task
backbone_kwargs = {"hidden_channels": 512, "num_layers": 4}
model = GraphNodeClassifier(
    num_features=datamodule.num_features, num_classes=datamodule.num_classes, backbone_kwargs=backbone_kwargs
)

# 3. Create the trainer and fit the model
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())
trainer.fit(model, datamodule=datamodule)

# 4. Classify some graphs!
predictions = model.predict(dataset[:3])
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("graph_node_classification_model.pt")
