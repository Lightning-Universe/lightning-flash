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
import os
from typing import List, Tuple

import pandas as pd
import torch

import flash
from flash.core.classification import SegmentationLabels
from flash.data.utils import download_data
from flash.vision import SemanticSegmentation, SemanticSegmentationData, SemanticSegmentationPreprocess

# 1. Download the data
# This is a Dataset with Semantic Segmentation Labels generated via CARLA self-driving simulator.
# The data was generated as part of the Lyft Udacity Challenge.
# More info here: https://www.kaggle.com/kumaresanmanickavelu/lyft-udacity-challenge
download_data(
    'https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180513A.zip', "data/"
)

# 2.1 Load the data


def load_data(data_root: str = 'data/') -> Tuple[List[str], List[str]]:
    images: List[str] = []
    labels: List[str] = []
    rgb_path = os.path.join(data_root, "CameraRGB")
    seg_path = os.path.join(data_root, "CameraSeg")
    for fname in os.listdir(rgb_path):
        images.append(os.path.join(rgb_path, fname))
        labels.append(os.path.join(seg_path, fname))
    return images, labels


images_filepaths, labels_filepaths = load_data()

# create the data module
datamodule = SemanticSegmentationData.from_filepaths(
    train_filepaths=images_filepaths,
    train_labels=labels_filepaths,
    batch_size=4,
    val_split=0.3,
    image_size=(300, 400),  # (600, 800)
)

# 2.2 Visualise the samples
labels_map = SegmentationLabels.create_random_labels_map(num_classes=21)
datamodule.set_map_labels(labels_map)
datamodule.show_train_batch("load_sample")
datamodule.show_train_batch("to_tensor_transform")

# 3. Build the model
model = SemanticSegmentation(
    backbone="torchvision/fcn_resnet50",
    num_classes=21,
)

# 4. Create the trainer.
trainer = flash.Trainer(
    max_epochs=20,
    gpus=1,
    # precision=16,  # why slower ? :)
)

# 5. Train the model
trainer.finetune(model, datamodule=datamodule, strategy='freeze')

# 6. Predict what's on a few images!
model.serializer = SegmentationLabels(labels_map, visualise=True)

predictions = model.predict([
    'data/CameraRGB/F61-1.png',
    'data/CameraRGB/F62-1.png',
    'data/CameraRGB/F63-1.png',
], datamodule.data_pipeline)

# 7. Save it!
trainer.save_checkpoint("semantic_segmentation_model.pt")
