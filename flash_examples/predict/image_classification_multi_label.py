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
from typing import Any

import torchvision.transforms.functional as T
from torchvision.utils import make_grid

from flash import Trainer
from flash.data.base_viz import BaseVisualization
from flash.data.utils import download_data
from flash.vision import ImageClassificationData, ImageClassifier

# 1. Download the data
# This is a subset of the movie poster genre prediction data set from the paper
# “Movie Genre Classification based on Poster Images with Deep Neural Networks” by Wei-Ta Chu and Hung-Jui Guo.
# Please consider citing their paper if you use it. More here: https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/
download_data("https://pl-flash-data.s3.amazonaws.com/movie_posters.zip", "data/")


# 2. Define our custom visualisation and datamodule
class CustomViz(BaseVisualization):

    def show_per_batch_transform(self, batch: Any, _) -> None:
        images = batch[0]["input"]
        image = make_grid(images, nrow=2)
        image = T.to_pil_image(image, 'RGB')
        image.show()


# 3. Load the model from a checkpoint
model = ImageClassifier.load_from_checkpoint(
    "https://flash-weights.s3.amazonaws.com/image_classification_multi_label_model.pt",
)

# 4a. Predict the genres of a few movie posters!
predictions = model.predict([
    "data/movie_posters/predict/tt0085318.jpg",
    "data/movie_posters/predict/tt0089461.jpg",
    "data/movie_posters/predict/tt0097179.jpg",
])
print(predictions)

# 4b. Or generate predictions with a whole folder!
datamodule = ImageClassificationData.from_folders(
    predict_folder="data/movie_posters/predict/",
    data_fetcher=CustomViz(),
    image_size=(128, 128),
)

predictions = Trainer().predict(model, datamodule=datamodule)
print(predictions)

# 5. Show some data (unless we're just testing)!
datamodule.show_predict_batch("per_batch_transform")
