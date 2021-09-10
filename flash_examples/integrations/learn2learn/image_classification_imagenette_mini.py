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
import warnings

import learn2learn as l2l
import torch

import flash
from flash.image import ImageClassificationData, ImageClassifier

warnings.simplefilter("ignore")


# reproduced from https://github.com/learnables/learn2learn/blob/master/examples/vision/protonet_miniimagenet.py#L154

# download MiniImagenet
train_dataset = l2l.vision.datasets.MiniImagenet(root="data", mode="train", download=True)
val_dataset = l2l.vision.datasets.MiniImagenet(root="data", mode="validation", download=True)
test_dataset = l2l.vision.datasets.MiniImagenet(root="data", mode="test", download=True)

# construct datamodule
datamodule = ImageClassificationData.from_tensors(
    train_data=train_dataset.x,
    train_targets=torch.from_numpy(train_dataset.y.astype(int)),
    val_data=val_dataset.x,
    val_targets=torch.from_numpy(val_dataset.y.astype(int)),
    test_data=test_dataset.x,
    test_targets=torch.from_numpy(test_dataset.y.astype(int)),
    num_workers=4,
)

ways = 30
model = ImageClassifier(
    ways,  # n
    backbone="resnet18",
    pretrained=True,
    training_strategy="prototypicalnetworks",
    optimizer=torch.optim.Adam,
    optimizer_kwargs={"lr": 0.001},
    training_strategy_kwargs={
        "epoch_length": 10 * 16,
        "meta_batch_size": 4,
        "num_tasks": 200,
        "test_num_tasks": 2000,
        "shots": 1,
        "test_ways": 5,
        "test_shots": 1,
        "test_queries": 15,
    },
)

trainer = flash.Trainer(
    max_epochs=200,
    gpus=4,
    accelerator="ddp",
    precision=16,
)
trainer.finetune(model, datamodule=datamodule, strategy="no_freeze")
