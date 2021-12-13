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
from flash.graph import GraphClassificationData, GraphClassifier

example_requires("graph")

from torch_geometric.datasets import TUDataset  # noqa: E402

# 1. Create the DataModule
dataset = TUDataset(root="data", name="KKI").shuffle()

datamodule = GraphClassificationData.from_datasets(
    train_dataset=dataset,
    val_split=0.1,
    batch_size=4,
)
# 2. Build the task
backbone_kwargs = {"hidden_channels": 512, "num_layers": 4}
model = GraphClassifier(
    num_features=datamodule.num_features, num_classes=datamodule.num_classes, backbone_kwargs=backbone_kwargs
)

# 3. Create the trainer and fit the model
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())
trainer.fit(model, datamodule=datamodule)

# 4. Classify some graphs!
datamodule = GraphClassificationData.from_datasets(
    predict_dataset=dataset[:3],
    batch_size=4,
)
predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("graph_classification_model.pt")
