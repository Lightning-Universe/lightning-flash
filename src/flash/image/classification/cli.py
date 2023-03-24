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

from flash.core.data.utils import download_data
from flash.core.utilities.flash_cli import FlashCLI
from flash.image import ImageClassificationData, ImageClassifier

__all__ = ["image_classification"]


def from_hymenoptera(
    batch_size: int = 4,
    num_workers: int = 0,
    **data_module_kwargs,
) -> ImageClassificationData:
    """Downloads and loads the Hymenoptera (Ants, Bees) data set."""
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "./data")
    return ImageClassificationData.from_folders(
        train_folder="data/hymenoptera_data/train/",
        val_folder="data/hymenoptera_data/val/",
        batch_size=batch_size,
        num_workers=num_workers,
        **data_module_kwargs,
    )


def from_movie_posters(
    batch_size: int = 4,
    num_workers: int = 0,
    **data_module_kwargs,
) -> ImageClassificationData:
    """Downloads and loads the movie posters genre classification data set."""
    download_data("https://pl-flash-data.s3.amazonaws.com/movie_posters.zip", "./data")

    def resolver(root, file_id):
        return os.path.join(root, f"{file_id}.jpg")

    return ImageClassificationData.from_csv(
        "Id",
        ["Action", "Romance", "Crime", "Thriller", "Adventure"],
        train_file="data/movie_posters/train/metadata.csv",
        train_resolver=resolver,
        val_file="data/movie_posters/val/metadata.csv",
        val_resolver=resolver,
        batch_size=batch_size,
        num_workers=num_workers,
        **data_module_kwargs,
    )


def image_classification():
    """Classify images."""
    cli = FlashCLI(
        ImageClassifier,
        ImageClassificationData,
        default_datamodule_builder=from_hymenoptera,
        additional_datamodule_builders=[from_movie_posters],
        default_arguments={
            "trainer.max_epochs": 3,
        },
        datamodule_attributes={"num_classes", "labels", "multi_label"},
    )

    cli.trainer.save_checkpoint("image_classification_model.pt")


if __name__ == "__main__":
    image_classification()
