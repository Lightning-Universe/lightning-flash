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
from flash.vision import ImageClassificationData, ImageClassifier
from flash.vision.classification.data import ImageClassificationPreprocess

# 1. Download the data
# This is a subset of the movie poster genre prediction data set from the paper
# “Movie Genre Classification based on Poster Images with Deep Neural Networks” by Wei-Ta Chu and Hung-Jui Guo.
# Please consider citing their paper if you use it. More here: https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/
download_data("https://pl-flash-data.s3.amazonaws.com/movie_posters.zip", "data/")

# 2. Load the data
genres = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy",
    "History", "Horror", "Music", "Musical", "Mystery", "N/A", "News", "Reality-TV", "Romance", "Sci-Fi", "Short",
    "Sport", "Thriller", "War", "Western"
]


def load_data(data: str, root: str = 'data/movie_posters') -> Tuple[List[str], List[List[int]]]:
    metadata = pd.read_csv(os.path.join(root, data, "metadata.csv"))

    images = []
    labels = []
    for _, row in metadata.iterrows():
        images.append(os.path.join(root, data, row['Id'] + ".jpg"))
        labels.append([int(row[genre]) for genre in genres])

    return images, labels


ImageClassificationPreprocess.image_size = (128, 128)

train_filepaths, train_labels = load_data('train')
val_filepaths, val_labels = load_data('val')
test_filepaths, test_labels = load_data('test')

datamodule = ImageClassificationData.from_filepaths(
    train_filepaths=train_filepaths,
    train_labels=train_labels,
    val_filepaths=val_filepaths,
    val_labels=val_labels,
    test_filepaths=test_filepaths,
    test_labels=test_labels,
    preprocess=ImageClassificationPreprocess(),
)

# 3. Build the model
model = ImageClassifier(
    backbone="resnet18",
    num_classes=len(genres),
    multi_label=True,
)

# 4. Create the trainer.
trainer = flash.Trainer(max_epochs=1, limit_train_batches=1, limit_val_batches=1)

# 5. Train the model
trainer.finetune(model, datamodule=datamodule, strategy=FreezeUnfreeze(unfreeze_epoch=1))

# 6a. Predict what's on a few images!

# Serialize predictions as labels.
model.serializer = Labels(genres, multi_label=True)

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
print(predictions)

# 7. Save it!
trainer.save_checkpoint("image_classification_multi_label_model.pt")
