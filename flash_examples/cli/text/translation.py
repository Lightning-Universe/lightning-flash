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

from flash.core.data.utils import download_data  # noqa: E402
from flash.core.utilities.flash_cli import FlashCLI  # noqa: E402
from flash.text import TranslationData, TranslationTask  # noqa: E402


def from_wmt_en_ro(
    backbone: str = "Helsinki-NLP/opus-mt-en-ro",
    batch_size: int = 4,
    num_workers: Optional[int] = None,
    **preprocess_kwargs,
) -> TranslationData:
    """Downloads and loads the WMT EN RO data set."""
    download_data("https://pl-flash-data.s3.amazonaws.com/wmt_en_ro.zip", "./data")
    return TranslationData.from_csv(
        "input",
        "target",
        train_file="data/wmt_en_ro/train.csv",
        val_file="data/wmt_en_ro/valid.csv",
        backbone=backbone,
        batch_size=batch_size,
        num_workers=num_workers,
        **preprocess_kwargs,
    )


# 1. Build the model, datamodule, and trainer. Expose them through CLI. Fine-tune
cli = FlashCLI(
    TranslationTask,
    TranslationData,
    default_datamodule_builder=from_wmt_en_ro,
    default_arguments={
        "trainer.max_epochs": 3,
        "model.backbone": "Helsinki-NLP/opus-mt-en-ro",
    }
)

# 2. Save the model!
cli.trainer.save_checkpoint("translation_model_en_ro.pt")
