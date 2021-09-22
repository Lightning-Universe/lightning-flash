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
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.transforms import ApplyToKeys
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageEmbedder
from flash.image.embedding.transforms import IMAGE_EMBEDDER_TRANSFORMS
from flash.image.embedding.vissl.transforms import simclr_collate_fn

# 1. Download the data and pre-process the data
transform = IMAGE_EMBEDDER_TRANSFORMS.get("simclr_transform")()

to_tensor_transform = ApplyToKeys(
    DefaultDataKeys.INPUT,
    transform,
)

datamodule = ImageClassificationData.from_datasets(
    train_dataset=CIFAR10(".", download=True),
    train_transform={
        "to_tensor_transform": to_tensor_transform,
        "collate": simclr_collate_fn,
    },
    batch_size=16,
)

# 2. Build the task
embedder = ImageEmbedder(
    backbone="resnet",
    training_strategy="barlow_twins",
    head="simclr_head",
    training_strategy_kwargs={"latent_embedding_dim": 128},
)

# 3. Create the trainer and pre-train the encoder
# use accelerator='ddp' when using GPU(s),
# i.e. flash.Trainer(max_epochs=3, gpus=1, accelerator='ddp')
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())
trainer.fit(embedder, datamodule=datamodule)

# 4. Save the model!
trainer.save_checkpoint("image_embedder_model.pt")

# 5. Download the downstream prediction dataset and generate embeddings
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "data/")

embeddings = embedder.predict(
    [
        "data/hymenoptera_data/predict/153783656_85f9c3ac70.jpg",
        "data/hymenoptera_data/predict/2039585088_c6f47c592e.jpg"
    ]
)
# list of embeddings for images sent to the predict function
print(embeddings)
