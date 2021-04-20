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
from typing import Any, Iterable, Optional

import pandas as pd
import torch
import torchvision.transforms.functional as T
from torchvision.utils import make_grid

from flash import Trainer
from flash.data.auto_dataset import AutoDataset
from flash.data.base_viz import BaseVisualization
from flash.data.utils import download_data
from flash.vision import ImageClassificationData, ImageClassifier
from flash.vision.classification.data import ImageClassificationPreprocess

# 1. Download the data
# This is a subset of the movie poster genre prediction data set from the paper
# “Movie Genre Classification based on Poster Images with Deep Neural Networks” by Wei-Ta Chu and Hung-Jui Guo.
# Please consider citing their paper if you use it. More here: https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/
download_data("https://pl-flash-data.s3.amazonaws.com/multi_label.zip", "data/")


# 2. Define our custom preprocess
class CustomMultiLabelPreprocess(ImageClassificationPreprocess):

    image_size = (128, 128)
    genres = [
        "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy",
        "History", "Horror", "Music", "Musical", "Mystery", "N/A", "News", "Reality-TV", "Romance", "Sci-Fi", "Short",
        "Sport", "Thriller", "War", "Western"
    ]

    def load_data(self, data: Any, dataset: Optional[AutoDataset] = None) -> Iterable:
        dataset.num_classes = len(self.genres)
        metadata = pd.read_csv(os.path.join(data, "metadata.csv"))

        images = []
        labels = []
        for _, row in metadata.iterrows():
            images.append(os.path.join(data, row['Id'] + ".jpg"))
            labels.append(torch.IntTensor([row[genre] for genre in self.genres]))

        return list(zip(images, labels))


# 3a. Define our custom visualisation and datamodule
class CustomViz(BaseVisualization):

    def show_per_batch_transform(self, batch: Any, _):
        images, labels = batch[0]
        image = make_grid(images, nrow=2)
        image = T.to_pil_image(image, 'RGB')
        image.show()


class CustomImageClassificationData(ImageClassificationData):

    @classmethod
    def configure_data_fetcher(cls):
        return CustomViz()


# 3b. Load the data
datamodule = CustomImageClassificationData.from_folders(
    train_folder="data/multi_label/train/",
    val_folder="data/multi_label/val/",
    test_folder="data/multi_label/test/",
    preprocess=CustomMultiLabelPreprocess(),
)

# 3c. Show some data!
datamodule.show_train_batch()

# 4. Load the model from a checkpoint
model = ImageClassifier.load_from_checkpoint("../finetuning/image_classification_multi_label_model.pt")

# 5a. Predict what's on a few images! ants or bees?
predictions = model.predict([
    "data/multi_label/val/tt0107111.jpg",
    "data/multi_label/val/tt0107199.jpg",
    "data/multi_label/val/tt0107606.jpg",
])
print(predictions)

# 5b. Or generate predictions with a whole folder!
datamodule = ImageClassificationData.from_folders(predict_folder="data/multi_label/predict/")

predictions = Trainer().predict(model, datamodule=datamodule)
print(predictions)
