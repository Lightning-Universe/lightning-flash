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

from flash.audio import AudioClassificationData
from flash.core.data.utils import download_data
from flash.core.utilities.flash_cli import FlashCLI
from flash.image import ImageClassifier

__all__ = ["audio_classification"]


def from_urban8k(
    batch_size: int = 4,
    **data_module_kwargs,
) -> AudioClassificationData:
    """Downloads and loads the Urban 8k sounds images data set."""
    download_data("https://pl-flash-data.s3.amazonaws.com/urban8k_images.zip", "./data")
    return AudioClassificationData.from_folders(
        train_folder="data/urban8k_images/train",
        val_folder="data/urban8k_images/val",
        batch_size=batch_size,
        **data_module_kwargs,
    )


def audio_classification():
    """Classify audio spectrograms."""
    cli = FlashCLI(
        ImageClassifier,
        AudioClassificationData,
        default_datamodule_builder=from_urban8k,
        default_arguments={
            "trainer.max_epochs": 3,
        },
        datamodule_attributes={"num_classes", "labels", "multi_label"},
    )

    cli.trainer.save_checkpoint("audio_classification_model.pt")


if __name__ == "__main__":
    audio_classification()
