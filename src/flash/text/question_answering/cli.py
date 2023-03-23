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
from flash.text import QuestionAnsweringData, QuestionAnsweringTask

__all__ = ["question_answering"]


def from_squad(
    batch_size: int = 4,
    **data_module_kwargs,
) -> QuestionAnsweringData:
    """Downloads and loads a tiny subset of the squad V2 data set."""
    download_data("https://pl-flash-data.s3.amazonaws.com/squad_tiny.zip", "./data/")

    return QuestionAnsweringData.from_squad_v2(
        train_file="./data/squad_tiny/train.json",
        val_file="./data/squad_tiny/val.json",
        batch_size=batch_size,
        **data_module_kwargs,
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
