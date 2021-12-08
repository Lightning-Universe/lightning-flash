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
from flash.text import SummarizationData, SummarizationTask

__all__ = ["summarization"]


def from_xsum(
    batch_size: int = 4,
    num_workers: int = 0,
    **input_transform_kwargs,
) -> SummarizationData:
    """Downloads and loads the XSum data set."""
    download_data("https://pl-flash-data.s3.amazonaws.com/xsum.zip", "./data/")
    return SummarizationData.from_csv(
        "input",
        "target",
        train_file="data/xsum/train.csv",
        val_file="data/xsum/valid.csv",
        batch_size=batch_size,
        num_workers=num_workers,
        **input_transform_kwargs,
    )


def summarization():
    """Summarize text."""
    cli = FlashCLI(
        SummarizationTask,
        SummarizationData,
        default_datamodule_builder=from_xsum,
        default_arguments={
            "trainer.max_epochs": 3,
            "model.backbone": "sshleifer/distilbart-xsum-1-1",
        },
    )

    cli.trainer.save_checkpoint("summarization_model_xsum.pt")


if __name__ == "__main__":
    summarization()
