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

import flash
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageClassifier

# 1. Create the DataModule
# Data set from the paper “Movie Genre Classification based on Poster Images with Deep Neural Networks”.
# More info here: https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/
download_data("https://pl-flash-data.s3.amazonaws.com/movie_posters.zip")
genres = ["Action", "Romance", "Crime", "Thriller", "Adventure"]


def load_data(data: str, root: str = 'data/movie_posters') -> Tuple[List[str], List[List[int]]]:
    metadata = pd.read_csv(osp.join(root, data, "metadata.csv"))
    return ([osp.join(root, data, row['Id'] + ".jpg") for _, row in metadata.iterrows()],
            [[int(row[genre]) for genre in genres] for _, row in metadata.iterrows()])


train_files, train_targets = load_data('train')
datamodule = ImageClassificationData.from_files(
    train_files=train_files,
    train_targets=train_targets,
    val_split=0.1,
    image_size=(128, 128),
)

# 2. Build the task
model = ImageClassifier(backbone="resnet18", num_classes=len(genres), multi_label=True)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Predict the genre of a few movies!
predictions = model.predict([
    "data/movie_posters/predict/tt0085318.jpg",
    "data/movie_posters/predict/tt0089461.jpg",
    "data/movie_posters/predict/tt0097179.jpg",
])
print(predictions)

# 7. Save the model!
trainer.save_checkpoint("image_classification_multi_label_model.pt")
