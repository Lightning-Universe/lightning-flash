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
from typing import Optional

from flash.core.data.utils import download_data
from flash.core.utilities.flash_cli import FlashCLI
from flash.video import VideoClassificationData, VideoClassifier

__all__ = ["video_classification"]


def from_kinetics(
    clip_sampler: str = "uniform",
    clip_duration: int = 1,
    decode_audio: bool = False,
    batch_size: int = 4,
    num_workers: Optional[int] = None,
    **preprocess_kwargs,
) -> VideoClassificationData:
    """Downloads and loads the Kinetics data set."""
    download_data("https://pl-flash-data.s3.amazonaws.com/kinetics.zip", "./data")
    return VideoClassificationData.from_folders(
        train_folder=os.path.join(os.getcwd(), "data/kinetics/train"),
        val_folder=os.path.join(os.getcwd(), "data/kinetics/val"),
        clip_sampler=clip_sampler,
        clip_duration=clip_duration,
        decode_audio=decode_audio,
        batch_size=batch_size,
        num_workers=num_workers,
        **preprocess_kwargs,
    )


def video_classification():
    """Classify videos."""
    cli = FlashCLI(
        VideoClassifier,
        VideoClassificationData,
        default_datamodule_builder=from_kinetics,
        default_arguments={
            "trainer.max_epochs": 3,
        },
    )

    cli.trainer.save_checkpoint("video_classification.pt")


if __name__ == "__main__":
    video_classification()
