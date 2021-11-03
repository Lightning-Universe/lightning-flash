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

# adapted from https://github.com/learnables/learn2learn/blob/master/examples/vision/protonet_miniimagenet.py#L154

import warnings

import kornia.augmentation as Ka
import kornia.geometry as Kg
import learn2learn as l2l
import torch
import torchvision
from torch import nn

import flash
from flash.core.data.io.input import InputDataKeys
from flash.core.data.transforms import ApplyToKeys, kornia_collate
from flash.image import ImageClassificationData, ImageClassifier

warnings.simplefilter("ignore")

# download MiniImagenet
train_dataset = l2l.vision.datasets.MiniImagenet(root="data", mode="train", download=True)
val_dataset = l2l.vision.datasets.MiniImagenet(root="data", mode="validation", download=True)
test_dataset = l2l.vision.datasets.MiniImagenet(root="data", mode="test", download=True)

train_transform = {
    "to_tensor_transform": nn.Sequential(
        ApplyToKeys(InputDataKeys.INPUT, torchvision.transforms.ToTensor()),
        ApplyToKeys(InputDataKeys.TARGET, torch.as_tensor),
    ),
    "post_tensor_transform": ApplyToKeys(
        InputDataKeys.INPUT,
        Kg.Resize((196, 196)),
        # SPATIAL
        Ka.RandomHorizontalFlip(p=0.25),
        Ka.RandomRotation(degrees=90.0, p=0.25),
        Ka.RandomAffine(degrees=1 * 5.0, shear=1 / 5, translate=1 / 20, p=0.25),
        Ka.RandomPerspective(distortion_scale=1 / 25, p=0.25),
        # PIXEL-LEVEL
        Ka.ColorJitter(brightness=1 / 30, p=0.25),  # brightness
        Ka.ColorJitter(saturation=1 / 30, p=0.25),  # saturation
        Ka.ColorJitter(contrast=1 / 30, p=0.25),  # contrast
        Ka.ColorJitter(hue=1 / 30, p=0.25),  # hue
        Ka.RandomMotionBlur(kernel_size=2 * (4 // 3) + 1, angle=1, direction=1.0, p=0.25),
        Ka.RandomErasing(scale=(1 / 100, 1 / 50), ratio=(1 / 20, 1), p=0.25),
    ),
    "collate": kornia_collate,
    "per_batch_transform_on_device": ApplyToKeys(
        InputDataKeys.INPUT,
        Ka.RandomHorizontalFlip(p=0.25),
    ),
}

# construct datamodule
datamodule = ImageClassificationData.from_tensors(
    train_data=train_dataset.x,
    train_targets=torch.from_numpy(train_dataset.y.astype(int)),
    val_data=val_dataset.x,
    val_targets=torch.from_numpy(val_dataset.y.astype(int)),
    test_data=test_dataset.x,
    test_targets=torch.from_numpy(test_dataset.y.astype(int)),
    num_workers=4,
    train_transform=train_transform,
)

model = ImageClassifier(
    backbone="resnet18",
    training_strategy="prototypicalnetworks",
    training_strategy_kwargs={
        "epoch_length": 10 * 16,
        "meta_batch_size": 4,
        "num_tasks": 200,
        "test_num_tasks": 2000,
        "ways": datamodule.num_classes,
        "shots": 1,
        "test_ways": 5,
        "test_shots": 1,
        "test_queries": 15,
    },
    optimizer=torch.optim.Adam,
    optimizer_kwargs={"lr": 0.001},
)

trainer = flash.Trainer(
    max_epochs=200,
    gpus=2,
    accelerator="ddp_shared",
    precision=16,
)
trainer.finetune(model, datamodule=datamodule, strategy="no_freeze")
