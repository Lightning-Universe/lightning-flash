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
import sys

sys.path.append("../../../")

from typing import Optional  # noqa: E402

from flash.core.data.utils import download_data  # noqa: E402
from flash.core.utilities.flash_cli import FlashCLI  # noqa: E402
from flash.image import ImageClassificationData, ImageClassifier  # noqa: E402


def from_movie_posters(
    batch_size: int = 4,
    num_workers: Optional[int] = None,
    **preprocess_kwargs,
) -> ImageClassificationData:
    """Downloads and loads the movie posters genre classification data set."""
    download_data("https://pl-flash-data.s3.amazonaws.com/movie_posters.zip", "./data")
    return ImageClassificationData.from_csv(
        "Id", ["Action", "Romance", "Crime", "Thriller", "Adventure"],
        train_file="data/movie_posters/train/metadata.csv",
        val_file="data/movie_posters/val/metadata.csv",
        batch_size=batch_size,
        num_workers=num_workers,
        **preprocess_kwargs
    )


# 1. Build the model, datamodule, and trainer. Expose them through CLI. Fine-tune
cli = FlashCLI(
    ImageClassifier,
    ImageClassificationData,
    default_datamodule_builder=from_movie_posters,
    default_arguments={
        "trainer.max_epochs": 3,
        "model.multi_label": True,
    }
)

# 2. Save the model!
cli.trainer.save_checkpoint("image_classification_multi_label_model.pt")
