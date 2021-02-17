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

from flash.core.data import download_data
from flash.vision import ImageEmbedder

# 1. Download the data
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "data/")

# 2. Create an ImageEmbedder with swav trained on imagenet.
# Check out SWAV: https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#swav
embedder = ImageEmbedder(backbone="swav-imagenet", embedding_dim=128)

# 3. Generate an embedding from an image path.
embeddings = embedder.predict(["data/hymenoptera_data/predict/153783656_85f9c3ac70.jpg"])

# 4. Print embeddings shape
print(embeddings.shape)

# 5. Create a tensor random image
random_image = torch.randn(1, 3, 32, 32)

# 6. Generate an embedding from this random image.
embeddings = embedder.predict(random_image)

# 7. Print embeddings shape
print(embeddings.shape)
