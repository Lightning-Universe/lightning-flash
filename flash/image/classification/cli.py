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
from typing import Optional

from flash.core.data.utils import download_data
from flash.core.utilities.flash_cli import FlashCLI
from flash.image import ImageClassificationData, ImageClassifier


def from_hymenoptera(
    batch_size: int = 4,
    num_workers: Optional[int] = None,
    **preprocess_kwargs,
) -> ImageClassificationData:
    """Downloads and loads the Hymenoptera (Ants, Bees) data set."""
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "./data")
    return ImageClassificationData.from_folders(
        train_folder="data/hymenoptera_data/train/",
        val_folder="data/hymenoptera_data/val/",
        batch_size=batch_size,
        num_workers=num_workers,
        **preprocess_kwargs,
    )


def main():
    cli = FlashCLI(
        ImageClassifier,
        ImageClassificationData,
        default_datamodule_builder=from_hymenoptera,
        default_arguments={
            'trainer.max_epochs': 3,
        }
    )

    cli.trainer.save_checkpoint("image_classification_model.pt")
