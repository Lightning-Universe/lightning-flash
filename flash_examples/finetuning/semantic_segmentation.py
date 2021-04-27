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
from flash.core.classification import Labels
from flash.core.finetuning import FreezeUnfreeze
from flash.data.utils import download_data
from flash.vision import SemanticSegmentation, SemanticSegmentationData, SemanticSegmentationPreprocess

# 1. Download the data
# This is a subset of the movie poster genre prediction data set from the paper
# “Movie Genre Classification based on Poster Images with Deep Neural Networks” by Wei-Ta Chu and Hung-Jui Guo.
# Please consider citing their paper if you use it. More here: https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/
#download_data("https://pl-flash-data.s3.amazonaws.com/movie_posters.zip", "data/")

# download from: https://www.kaggle.com/kumaresanmanickavelu/lyft-udacity-challenge

# 2. Load the data
num_classes: int = 21

labels_map = {}
for i in range(num_classes):
    labels_map[i] = torch.randint(0, 255, (3, ))

root_dir = '/home/edgar/data/archive'
datasets = ['dataA', 'dataB', 'dataC', 'dataD', 'dataE']


def load_data(data_root: str, datasets: List[str]) -> Tuple[List[str], List[str]]:
    images: List[str] = []
    labels: List[str] = []
    for data in datasets:
        data_dir = os.path.join(root_dir, data, data)
        rgb_path = os.path.join(data_dir, "CameraRGB")
        seg_path = os.path.join(data_dir, "CameraSeg")
        for fname in os.listdir(rgb_path):
            images.append(os.path.join(rgb_path, fname))
            labels.append(os.path.join(seg_path, fname))
    return images, labels


train_filepaths, train_labels = load_data(root_dir, datasets[:3])
val_filepaths, val_labels = load_data(root_dir, datasets[3:4])
predict_filepaths, predict_labels = load_data(root_dir, datasets[4:5])

datamodule = SemanticSegmentationData.from_filepaths(
    train_filepaths=train_filepaths,
    train_labels=train_labels,
    val_filepaths=val_filepaths,
    val_labels=val_labels,
    batch_size=4,
    image_size=(300, 400),  # (600, 800)
)
datamodule.set_map_labels(labels_map)
datamodule.show_train_batch("load_sample")
datamodule.show_train_batch("to_tensor_transform")

# 3. Build the model
model = SemanticSegmentation(
    backbone="torchvision/fcn_resnet50",
    num_classes=num_classes,
)

# 4. Create the trainer.
#trainer = flash.Trainer(max_epochs=5, limit_train_batches=1, limit_val_batches=1)
trainer = flash.Trainer(
    max_epochs=20,
    gpus=1,
    #precision=16,  # why slower ? :)
)

# 5. Train the model
trainer.finetune(model, datamodule=datamodule, strategy='freeze')
# TODO: getting error: BrokenPipeError: [Errno 32] Broken pipe

# 6. Predict what's on a few images!

import kornia as K
import matplotlib.pyplot as plt

from flash.data.process import ProcessState, Serializer


class SegmentationLabels(Serializer):

    def __init__(self, map_labels, visualise):
        super().__init__()
        self.map_labels = map_labels
        self.visualise = visualise

    def _labels_to_image(self, img_labels: torch.Tensor) -> torch.Tensor:
        assert len(img_labels.shape) == 2, img_labels.shape
        H, W = img_labels.shape
        out = torch.empty(3, H, W, dtype=torch.uint8)
        for label_id, label_val in self.map_labels.items():
            mask = (img_labels == label_id)
            for i in range(3):
                out[i].masked_fill_(mask, label_val[i])
        return out

    def serialize(self, sample: torch.Tensor) -> torch.Tensor:
        assert len(sample.shape) == 3, sample.shape
        labels = torch.argmax(sample, dim=-3)  # HxW
        if self.visualise:
            labels_vis = self._labels_to_image(labels)
            labels_vis = K.utils.tensor_to_image(labels_vis)
            plt.imshow(labels_vis)
            plt.show()
        return labels


model.serializer = SegmentationLabels(labels_map, visualise=True)

predictions = model.predict([
    predict_filepaths[0],
    predict_filepaths[1],
    predict_filepaths[2],
], datamodule.data_pipeline)

#print(predictions)

# 7. Save it!
trainer.save_checkpoint("semantic_segmentation_model.pt")
