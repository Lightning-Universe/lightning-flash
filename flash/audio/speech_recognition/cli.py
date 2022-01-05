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

from flash.audio import SpeechRecognition, SpeechRecognitionData
from flash.core.data.utils import download_data
from flash.core.utilities.flash_cli import FlashCLI

__all__ = ["speech_recognition"]


def from_timit(
    val_split: float = 0.1,
    batch_size: int = 4,
    num_workers: int = 0,
    **input_transform_kwargs,
) -> SpeechRecognitionData:
    """Downloads and loads the timit data set."""
    download_data("https://pl-flash-data.s3.amazonaws.com/timit_data.zip", "./data")
    return SpeechRecognitionData.from_json(
        "file",
        "text",
        train_file="data/timit/train.json",
        test_file="data/timit/test.json",
        val_split=val_split,
        batch_size=batch_size,
        num_workers=num_workers,
        **input_transform_kwargs,
    )


def speech_recognition():
    """Speech recognition."""
    cli = FlashCLI(
        SpeechRecognition,
        SpeechRecognitionData,
        default_datamodule_builder=from_timit,
        default_arguments={
            "trainer.max_epochs": 3,
        },
        finetune=False,
    )

    cli.trainer.save_checkpoint("speech_recognition_model.pt")


if __name__ == "__main__":
    speech_recognition()
