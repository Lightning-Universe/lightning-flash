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

from flash.audio import AudioClassificationData  # noqa: E402
from flash.core.data.utils import download_data  # noqa: E402
from flash.core.utilities.flash_cli import FlashCLI  # noqa: E402
from flash.image import ImageClassifier  # noqa: E402


def from_urban8k(
    batch_size: int = 4,
    num_workers: Optional[int] = None,
    **preprocess_kwargs,
) -> AudioClassificationData:
    """Downloads and loads the Urban 8k sounds images data set."""
    download_data("https://pl-flash-data.s3.amazonaws.com/urban8k_images.zip", "./data")
    return AudioClassificationData.from_folders(
        train_folder="data/urban8k_images/train",
        val_folder="data/urban8k_images/val",
        batch_size=batch_size,
        num_workers=num_workers,
        **preprocess_kwargs,
    )


# 1. Build the model, datamodule, and trainer. Expose them through CLI. Fine-tune
cli = FlashCLI(
    ImageClassifier,
    AudioClassificationData,
    default_datamodule_builder=from_urban8k,
    default_arguments={
        'trainer.max_epochs': 3,
    }
)

# 2. Save the model!
cli.trainer.save_checkpoint("audio_classification_model.pt")
