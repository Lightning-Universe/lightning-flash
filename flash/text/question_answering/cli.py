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
from typing import Optional

from flash.core.data.utils import download_data
from flash.core.utilities.flash_cli import FlashCLI
from flash.text import QuestionAnsweringData, QuestionAnsweringTask

__all__ = ["question_answering"]


def from_squad(
    backbone: str = "distilbert-base-uncased",
    batch_size: int = 4,
    num_workers: Optional[int] = None,
    **preprocess_kwargs,
) -> QuestionAnsweringData:
    """Downloads and loads the XSum data set."""
    download_data("https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json", "./data/")
    download_data("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json", "./data/")
    return QuestionAnsweringData.from_squad_v2(
        train_file="./data/train-v2.0.json",
        val_file="./data/dev-v2.0.json",
        backbone=backbone,
        batch_size=batch_size,
        num_workers=num_workers,
        **preprocess_kwargs,
    )


def question_answering():
    """Extractive Question Answering."""
    cli = FlashCLI(
        QuestionAnsweringTask,
        QuestionAnsweringData,
        default_datamodule_builder=from_squad,
        default_arguments={
            "trainer.max_epochs": 3,
            "model.backbone": "distilbert-base-uncased",
        },
    )

    cli.trainer.save_checkpoint("question_answering_model.pt")


if __name__ == "__main__":
    question_answering()
