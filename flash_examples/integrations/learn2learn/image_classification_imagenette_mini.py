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
import learn2learn as l2l
import torch

import flash
from flash.image import ImageClassificationData, ImageClassifier

# download MiniImagenet
train_dataset = l2l.vision.datasets.MiniImagenet(root="./data", mode="train", download=True)
val_dataset = l2l.vision.datasets.MiniImagenet(root="./data", mode="validation", download=True)
test_dataset = l2l.vision.datasets.MiniImagenet(root="./data", mode="test", download=True)

# construct datamodule
datamodule = ImageClassificationData.from_tensors(
    # NOTE: they return tensors for x but arrays for y -> I must manually convert it
    train_data=train_dataset.x,
    train_targets=torch.from_numpy(train_dataset.y.astype(int)),
    val_data=val_dataset.x,
    val_targets=torch.from_numpy(val_dataset.y.astype(int)),
    test_data=test_dataset.x,
    test_targets=torch.from_numpy(test_dataset.y.astype(int)),
)

model = ImageClassifier(
    64,  # NOTE: from_tensors apparently does not compute the num_classes automatically
    backbone="resnet18",
    training_strategy="prototypicalnetworks",
    training_strategy_kwargs={"shots": 4, "meta_batch_size": 10},
)

trainer = flash.Trainer(fast_dev_run=True)
trainer.finetune(model, datamodule=datamodule, strategy="no_freeze")
