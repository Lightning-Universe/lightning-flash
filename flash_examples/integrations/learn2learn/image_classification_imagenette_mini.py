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

import kornia.augmentation as Ka
import kornia.geometry as Kg
import learn2learn as l2l
import torch
import torchvision
from torch import nn

import flash
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.transforms import ApplyToKeys, kornia_collate
from flash.image import ImageClassificationData, ImageClassifier

warnings.simplefilter("ignore")


# adapted from https://github.com/learnables/learn2learn/blob/master/examples/vision/protonet_miniimagenet.py#L154
# download MiniImagenet
train_dataset = l2l.vision.datasets.MiniImagenet(root="data", mode="train", download=True)
val_dataset = l2l.vision.datasets.MiniImagenet(root="data", mode="validation", download=True)
test_dataset = l2l.vision.datasets.MiniImagenet(root="data", mode="test", download=True)

train_transform = {
    "to_tensor_transform": nn.Sequential(
        ApplyToKeys(DefaultDataKeys.INPUT, torchvision.transforms.ToTensor()),
        ApplyToKeys(DefaultDataKeys.TARGET, torch.as_tensor),
    ),
    "post_tensor_transform": ApplyToKeys(
        DefaultDataKeys.INPUT,
        Kg.Resize((196, 196)),
        # SPATIAL
        Ka.RandomHorizontalFlip(p=1),
        Ka.RandomRotation(degrees=90.0, p=1),
        Ka.RandomAffine(degrees=4 * 5.0, shear=4 / 5, translate=4 / 20, p=1),
        Ka.RandomPerspective(distortion_scale=4 / 25, p=1),
        # PIXEL-LEVEL
        Ka.ColorJitter(brightness=4 / 30, p=1),  # brightness
        Ka.ColorJitter(saturation=4 / 30, p=1),  # saturation
        Ka.ColorJitter(contrast=4 / 30, p=1),  # contrast
        Ka.ColorJitter(hue=4 / 30, p=1),  # hue
        Ka.ColorJitter(p=0),  # identity
        Ka.RandomMotionBlur(kernel_size=2 * (4 // 3) + 1, angle=4, direction=1.0, p=1),
        Ka.RandomErasing(scale=(4 / 100, 4 / 50), ratio=(4 / 20, 4), p=1),
    ),
    "collate": kornia_collate,
    "per_batch_transform_on_device": ApplyToKeys(
        DefaultDataKeys.INPUT,
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
    datamodule.num_classes,  # ways
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
