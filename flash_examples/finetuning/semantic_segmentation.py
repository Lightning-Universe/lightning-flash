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
from flash.vision import SemanticSegmentation, SemanticSegmentationData

# 1. Download the data
# This is a subset of the movie poster genre prediction data set from the paper
# “Movie Genre Classification based on Poster Images with Deep Neural Networks” by Wei-Ta Chu and Hung-Jui Guo.
# Please consider citing their paper if you use it. More here: https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/
#download_data("https://pl-flash-data.s3.amazonaws.com/movie_posters.zip", "data/")

# 2. Load the data
# TODO: define labels maps
num_classes = 21

labels_map = {}
for i in range(num_classes):
    labels_map[i] = torch.randint(0, 255, (3, ))

ROOT_DIR = '/home/edgar/data/archive/dataA/dataA'


def load_data(data: str, root: str = '') -> Tuple[List[str], List[str]]:
    images: List[str] = []
    labels: List[str] = []

    rgb_path = os.path.join(ROOT_DIR, "CameraRGB")
    seg_path = os.path.join(ROOT_DIR, "CameraSeg")

    for fname in os.listdir(rgb_path):
        images.append(os.path.join(rgb_path, fname))
        labels.append(os.path.join(seg_path, fname))

    return images, labels


train_filepaths, train_labels = load_data('train')
val_filepaths, val_labels = load_data('val')
test_filepaths, test_labels = load_data('test')

datamodule = SemanticSegmentationData.from_filepaths(
    train_filepaths=train_filepaths,
    train_labels=train_labels,
    val_filepaths=val_filepaths,
    val_labels=val_labels,
    test_filepaths=test_filepaths,
    test_labels=test_labels,
    batch_size=16
    #preprocess=ImageClassificationPreprocess(),
)
datamodule.set_map_labels(labels_map)
'''datamodule.set_block_viz_window(False)
datamodule.show_train_batch("load_sample")
datamodule.set_block_viz_window(True)'''
datamodule.show_train_batch("load_sample")
datamodule.show_train_batch("to_tensor_transform")

# 3. Build the model
model = SemanticSegmentation(
    backbone="torchvision/fcn_resnet50",
    num_classes=num_classes,
)

# 4. Create the trainer.
trainer = flash.Trainer(max_epochs=1, limit_train_batches=1, limit_val_batches=1)

# 5. Train the model
trainer.finetune(model, datamodule=datamodule, strategy=FreezeUnfreeze(unfreeze_epoch=1))

# 6a. Predict what's on a few images!

# Serialize predictions as labels.
'''model.serializer = Labels(genres, multi_label=True)

predictions = model.predict([
    "data/movie_posters/val/tt0361500.jpg",
    "data/movie_posters/val/tt0361748.jpg",
    "data/movie_posters/val/tt0362478.jpg",
])

print(predictions)

datamodule = ImageClassificationData.from_folders(
    predict_folder="data/movie_posters/predict/",
    preprocess=model.preprocess,
)

# 6b. Or generate predictions with a whole folder!
predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)'''

# 7. Save it!
trainer.save_checkpoint("semantic_segmentation_model.pt")
