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
from typing import Any

from flash import Trainer
from flash.data.base_viz import BaseVisualization
from flash.data.utils import download_data
from flash.vision import ImageClassificationData, ImageClassifier

import torchvision.transforms.functional as T
from torchvision.utils import make_grid

# 1. Download the data
# This is a subset of the movie poster genre prediction data set from the paper
# “Movie Genre Classification based on Poster Images with Deep Neural Networks” by Wei-Ta Chu and Hung-Jui Guo.
# Please consider citing their paper if you use it. More here: https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/
download_data("https://pl-flash-data.s3.amazonaws.com/movie_posters.zip", "data/")


# 2. Define our custom visualisation and datamodule
class CustomViz(BaseVisualization):

    def show_per_batch_transform(self, batch: Any, _) -> None:
        images = batch[0]
        image = make_grid(images, nrow=2)
        image = T.to_pil_image(image, 'RGB')
        image.show()


# 3. Load the model from a checkpoint
model = ImageClassifier.load_from_checkpoint(
    "https://flash-weights.s3.amazonaws.com/image_classification_multi_label_model.pt",
)

# 4a. Predict the genres of a few movie posters!
predictions = model.predict([
    "data/movie_posters/val/tt0086873.jpg",
    "data/movie_posters/val/tt0088247.jpg",
    "data/movie_posters/val/tt0088930.jpg",
])
print(predictions)

# 4b. Or generate predictions with a whole folder!
datamodule = ImageClassificationData.from_folders(
    predict_folder="data/movie_posters/predict/",
    data_fetcher=CustomViz(),
    preprocess=model.preprocess,
)

predictions = Trainer().predict(model, datamodule=datamodule)
print(predictions)

# 5. Show some data!
datamodule.show_predict_batch("per_batch_transform")
