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
import os.path as osp
from typing import List, Tuple

import pandas as pd
from torchmetrics import F1

import flash
from flash.core.classification import Labels
from flash.data.utils import download_data
from flash.vision import ImageClassificationData, ImageClassifier
from flash.vision.classification.data import ImageClassificationPreprocess

# 1. Download the data
# This is a subset of the movie poster genre prediction data set from the paper
# “Movie Genre Classification based on Poster Images with Deep Neural Networks” by Wei-Ta Chu and Hung-Jui Guo.
# Please consider citing their paper if you use it. More here: https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/
download_data("https://pl-flash-data.s3.amazonaws.com/movie_posters.zip", "data/")

# 2. Load the data
genres = ["Action", "Animation", "Comedy", "Horror", "Musical"]


def load_data(data: str, root: str = 'data/movie_posters') -> Tuple[List[str], List[List[int]]]:
    metadata = pd.read_csv(osp.join(root, data, "metadata.csv"))
    return ([osp.join(root, data, row['Id'] + ".jpg") for _, row in metadata.iterrows()],
            [[int(row[genre]) for genre in genres] for _, row in metadata.iterrows()])


train_filepaths, train_labels = load_data('train')
test_filepaths, test_labels = load_data('test')

datamodule = ImageClassificationData.from_filepaths(
    train_filepaths=train_filepaths,
    train_labels=train_labels,
    test_filepaths=test_filepaths,
    test_labels=test_labels,
    preprocess=ImageClassificationPreprocess(image_size=(128, 128)),
    val_split=0.1,  # Use 10 % of the train dataset to generate validation one.
)

# 3. Build the model
model = ImageClassifier(
    backbone="resnet18",
    num_classes=len(genres),
    multi_label=True,
    metrics=F1(num_classes=len(genres)),
    scheduler="linear_schedule_with_warmup",
    scheduler_kwargs={"num_warmup_steps": 0.1}
)

# 4. Create the trainer. Train on 2 gpus for 10 epochs.
trainer = flash.Trainer(max_epochs=10, gpus=2, accelerator='ddp')

# 5. Train the model
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 6a. Predict what's on a few images!

# Serialize predictions as labels.
model.serializer = Labels(genres, multi_label=True)

predictions = model.predict([
    "data/movie_posters/val/tt0085995.jpg",
    "data/movie_posters/val/tt0086508.jpg",
    "data/movie_posters/val/tt0088184.jpg",
])
