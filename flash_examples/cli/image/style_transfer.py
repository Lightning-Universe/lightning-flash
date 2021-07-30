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
import sys

sys.path.append("../../../")

from typing import Optional  # noqa: E402

import flash  # noqa: E402
from flash.core.data.utils import download_data  # noqa: E402
from flash.core.utilities.flash_cli import FlashCLI  # noqa: E402
from flash.image import StyleTransfer, StyleTransferData  # noqa: E402


def from_coco_128(
    batch_size: int = 4,
    num_workers: Optional[int] = None,
    **preprocess_kwargs,
) -> StyleTransferData:
    """Downloads and loads the COCO 128 data set."""
    download_data("https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip", "data/")
    return StyleTransferData.from_folders(
        train_folder="data/coco128/images/train2017/",
        batch_size=batch_size,
        num_workers=num_workers,
        **preprocess_kwargs
    )


# 1. Build the model, datamodule, and trainer. Expose them through CLI. Fine-tune
cli = FlashCLI(
    StyleTransfer,
    StyleTransferData,
    default_datamodule_builder=from_coco_128,
    default_arguments={
        "trainer.max_epochs": 3,
        "model.style_image": os.path.join(flash.ASSETS_ROOT, "starry_night.jpg")
    },
    finetune=False,
)

# 2. Save the model!
cli.trainer.save_checkpoint("style_transfer_model.pt")
