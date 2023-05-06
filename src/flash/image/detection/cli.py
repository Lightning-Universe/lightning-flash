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
from typing import Any, Dict, Optional

from flash.core.data.utils import download_data
from flash.core.utilities.flash_cli import FlashCLI
from flash.image import ObjectDetectionData, ObjectDetector

__all__ = ["object_detection"]


def from_coco_128(
    val_split: float = 0.1,
    transform_kwargs: Optional[Dict[str, Any]] = None,
    batch_size: int = 1,
    **data_module_kwargs,
) -> ObjectDetectionData:
    """Downloads and loads the COCO 128 data set."""
    download_data("https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip", "data/")
    return ObjectDetectionData.from_coco(
        train_folder="data/coco128/images/train2017/",
        train_ann_file="data/coco128/annotations/instances_train2017.json",
        val_split=val_split,
        transform_kwargs=dict(image_size=(128, 128)) if transform_kwargs is None else transform_kwargs,
        batch_size=batch_size,
        **data_module_kwargs,
    )


def object_detection():
    """Detect objects in images."""
    cli = FlashCLI(
        ObjectDetector,
        ObjectDetectionData,
        default_datamodule_builder=from_coco_128,
        default_arguments={
            "trainer.max_epochs": 3,
        },
    )

    cli.trainer.save_checkpoint("object_detection_model.pt")


if __name__ == "__main__":
    object_detection()
