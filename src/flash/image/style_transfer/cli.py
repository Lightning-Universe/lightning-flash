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

import flash
from flash.core.data.utils import download_data
from flash.core.utilities.flash_cli import FlashCLI
from flash.image import StyleTransfer, StyleTransferData

__all__ = ["style_transfer"]


def from_coco_128(
    batch_size: int = 4,
    num_workers: int = 0,
    **data_module_kwargs,
) -> StyleTransferData:
    """Downloads and loads the COCO 128 data set."""
    download_data("https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip", "data/")
    return StyleTransferData.from_folders(
        train_folder="data/coco128/images/train2017/",
        batch_size=batch_size,
        num_workers=num_workers,
        **data_module_kwargs,
    )


def style_transfer():
    """Image style transfer."""
    cli = FlashCLI(
        StyleTransfer,
        StyleTransferData,
        default_datamodule_builder=from_coco_128,
        default_arguments={
            "trainer.max_epochs": 3,
            "model.style_image": os.path.join(flash.ASSETS_ROOT, "starry_night.jpg"),
        },
        finetune=False,
    )

    cli.trainer.save_checkpoint("style_transfer_model.pt")


if __name__ == "__main__":
    style_transfer()
