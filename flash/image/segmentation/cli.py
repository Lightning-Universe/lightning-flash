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

from flash.core.data.utils import download_data
from flash.core.utilities.flash_cli import FlashCLI
from flash.image import SemanticSegmentation, SemanticSegmentationData

__all__ = ["semantic_segmentation"]


def from_carla(
    num_classes: int = 21,
    val_split: float = 0.1,
    batch_size: int = 4,
    **data_module_kwargs,
) -> SemanticSegmentationData:
    """Downloads and loads the CARLA capture data set."""
    download_data(
        "https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180513A.zip",
        "./data",
    )
    return SemanticSegmentationData.from_folders(
        train_folder="data/CameraRGB",
        train_target_folder="data/CameraSeg",
        val_split=val_split,
        batch_size=batch_size,
        num_classes=num_classes,
        **data_module_kwargs,
    )


def semantic_segmentation():
    """Segment objects in images."""
    cli = FlashCLI(
        SemanticSegmentation,
        SemanticSegmentationData,
        default_datamodule_builder=from_carla,
        default_arguments={
            "trainer.max_epochs": 3,
        },
    )

    cli.trainer.save_checkpoint("semantic_segmentation_model.pt")


if __name__ == "__main__":
    semantic_segmentation()
