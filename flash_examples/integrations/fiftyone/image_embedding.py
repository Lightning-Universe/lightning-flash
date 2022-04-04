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
import fiftyone as fo
import fiftyone.brain as fob
import numpy as np

import torch

import flash
from flash.core.data.utils import download_data
from flash.image import ImageEmbedder
from flash.image.classification.data import ImageClassificationData

# 1 Download data
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip")

# 2 Load data into FiftyOne
dataset = fo.Dataset.from_dir(
    "data/hymenoptera_data/test/",
    fo.types.ImageClassificationDirectoryTree,
)
datamodule = ImageClassificationData.from_files(
    predict_files=dataset.values("filepath"),
    batch_size=16,
)

# 3 Load model
embedder = ImageEmbedder(backbone="resnet18")

# 4 Generate embeddings
trainer = flash.Trainer(gpus=torch.cuda.device_count())
embedding_batches = trainer.predict(embedder, datamodule=datamodule)
embeddings = np.stack(sum(embedding_batches, []))

# 5 Visualize in FiftyOne App
results = fob.compute_visualization(dataset, embeddings=embeddings)
session = fo.launch_app(dataset)
plot = results.visualize(labels="ground_truth.label")
plot.show()

# Optional: block execution until App is closed
session.wait()
