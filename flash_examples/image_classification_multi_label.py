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
import flash
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageClassifier

# 1. Create the DataModule
# Data set from the paper “Movie Genre Classification based on Poster Images with Deep Neural Networks”.
# More info here: https://www.cs.ccu.edu.tw/~wtchu/projects/MoviePoster/
download_data("https://pl-flash-data.s3.amazonaws.com/movie_posters.zip")

datamodule = ImageClassificationData.from_csv(
    "Id",
    ["Action", "Romance", "Crime", "Thriller", "Adventure"],
    train_file="data/movie_posters/train/metadata.csv",
    val_file="data/movie_posters/val/metadata.csv",
    image_size=(128, 128),
)

# 2. Build the task
model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes, multi_label=True)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Predict the genre of a few movies!
predictions = model.predict(
    [
        "data/movie_posters/predict/tt0085318.jpg",
        "data/movie_posters/predict/tt0089461.jpg",
        "data/movie_posters/predict/tt0097179.jpg",
    ]
)
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("image_classification_multi_label_model.pt")
