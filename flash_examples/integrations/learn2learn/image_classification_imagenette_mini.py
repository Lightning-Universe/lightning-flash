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

"""## Train file https://www.dropbox.com/s/9g8c6w345s2ek03/mini-imagenet-cache-train.pkl?dl=1

## Validation File
https://www.dropbox.com/s/ip1b7se3gij3r1b/mini-imagenet-cache-validation.pkl?dl=1

Followed by renaming the pickle files
cp './mini-imagenet-cache-train.pkl?dl=1' './mini-imagenet-cache-train.pkl'
cp './mini-imagenet-cache-validation.pkl?dl=1' './mini-imagenet-cache-validation.pkl'
"""

import warnings
from dataclasses import dataclass
from typing import Tuple, Union

import kornia.augmentation as Ka
import kornia.geometry as Kg
import learn2learn as l2l
import torch
import torchvision.transforms as T

import flash
from flash.core.data.io.input import DataKeys
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.transforms import ApplyToKeys
from flash.image import ImageClassificationData, ImageClassifier

warnings.simplefilter("ignore")

# download MiniImagenet
train_dataset = l2l.vision.datasets.MiniImagenet(root="./", mode="train", download=False)
val_dataset = l2l.vision.datasets.MiniImagenet(root="./", mode="validation", download=False)


@dataclass
class ImageClassificationInputTransform(InputTransform):

    image_size: Tuple[int, int] = (196, 196)
    mean: Union[float, Tuple[float, float, float]] = (0.485, 0.456, 0.406)
    std: Union[float, Tuple[float, float, float]] = (0.229, 0.224, 0.225)

    def per_sample_transform(self):
        return T.Compose(
            [
                ApplyToKeys(
                    DataKeys.INPUT,
                    T.Compose(
                        [
                            T.ToTensor(),
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
                        ]
                    ),
                ),
                ApplyToKeys(DataKeys.TARGET, torch.as_tensor),
            ]
        )

    def train_per_sample_transform(self):
        return T.Compose(
            [
                ApplyToKeys(
                    DataKeys.INPUT,
                    T.Compose(
                        [
                            T.ToTensor(),
                            T.Resize(self.image_size),
                            T.Normalize(self.mean, self.std),
                            T.RandomHorizontalFlip(),
                            T.ColorJitter(),
                            T.RandomAutocontrast(),
                            T.RandomPerspective(),
                        ]
                    ),
                ),
                ApplyToKeys("target", torch.as_tensor),
            ]
        )

    def per_batch_transform_on_device(self):
        return ApplyToKeys(
            DataKeys.INPUT,
            Ka.RandomHorizontalFlip(p=0.25),
        )


# construct datamodule

datamodule = ImageClassificationData.from_tensors(
    train_data=train_dataset.x,
    train_targets=torch.from_numpy(train_dataset.y.astype(int)),
    val_data=val_dataset.x,
    val_targets=torch.from_numpy(val_dataset.y.astype(int)),
    train_transform=ImageClassificationInputTransform,
    val_transform=ImageClassificationInputTransform,
    batch_size=1,
)

model = ImageClassifier(
    backbone="resnet18",
    training_strategy="prototypicalnetworks",
    training_strategy_kwargs={
        "epoch_length": 10 * 16,
        "meta_batch_size": 1,
        "num_tasks": 200,
        "test_num_tasks": 2000,
        "ways": datamodule.num_classes,
        "shots": 1,
        "test_ways": 5,
        "test_shots": 1,
        "test_queries": 15,
    },
    optimizer=torch.optim.Adam,
    learning_rate=0.001,
)

trainer = flash.Trainer(
    max_epochs=1,
    gpus=1,
    accelerator="gpu",
    precision=16,
)

trainer.finetune(model, datamodule=datamodule, strategy="no_freeze")
