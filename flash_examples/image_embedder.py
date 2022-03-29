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
import torch
from torchvision.datasets import CIFAR10

import flash
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageEmbedder

# 1. Download the data and prepare the datamodule
datamodule = ImageClassificationData.from_datasets(
    train_dataset=CIFAR10(".", download=True),
    batch_size=4,
)

# 2. Build the task
embedder = ImageEmbedder(
    backbone="vision_transformer",
    training_strategy="barlow_twins",
    head="barlow_twins_head",
    pretraining_transform="barlow_twins_transform",
    training_strategy_kwargs={"latent_embedding_dim": 128},
    pretraining_transform_kwargs={"size_crops": [32]},
)

# 3. Create the trainer and pre-train the encoder
trainer = flash.Trainer(max_epochs=1, gpus=torch.cuda.device_count())
trainer.fit(embedder, datamodule=datamodule)

# 4. Save the model!
trainer.save_checkpoint("image_embedder_model.pt")

# 5. Download the downstream prediction dataset and generate embeddings
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "data/")

datamodule = ImageClassificationData.from_files(
    predict_files=[
        "data/hymenoptera_data/predict/153783656_85f9c3ac70.jpg",
        "data/hymenoptera_data/predict/2039585088_c6f47c592e.jpg",
    ],
    batch_size=3,
)
embeddings = trainer.predict(embedder, datamodule=datamodule)

# list of embeddings for images sent to the predict function
print(embeddings)
