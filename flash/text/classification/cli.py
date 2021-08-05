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
from flash.text import TextClassificationData, TextClassifier

__all__ = ["text_classification"]


def from_imdb(
    backbone: str = "prajjwal1/bert-medium",
    batch_size: int = 4,
    num_workers: Optional[int] = None,
    **preprocess_kwargs,
) -> TextClassificationData:
    """Downloads and loads the IMDB sentiment classification data set."""
    download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", "./data/")
    return TextClassificationData.from_csv(
        "review",
        "sentiment",
        train_file="data/imdb/train.csv",
        val_file="data/imdb/valid.csv",
        backbone=backbone,
        batch_size=batch_size,
        num_workers=num_workers,
        **preprocess_kwargs,
    )


def from_toxic(
    backbone: str = "unitary/toxic-bert",
    val_split: float = 0.1,
    batch_size: int = 4,
    num_workers: Optional[int] = None,
    **preprocess_kwargs,
) -> TextClassificationData:
    """Downloads and loads the Jigsaw toxic comments data set."""
    download_data("https://pl-flash-data.s3.amazonaws.com/jigsaw_toxic_comments.zip", "./data")
    return TextClassificationData.from_csv(
        "comment_text",
        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
        train_file="data/jigsaw_toxic_comments/train.csv",
        backbone=backbone,
        val_split=val_split,
        batch_size=batch_size,
        num_workers=num_workers,
        **preprocess_kwargs,
    )


def text_classification():
    """Classify text."""
    cli = FlashCLI(
        TextClassifier,
        TextClassificationData,
        default_datamodule_builder=from_imdb,
        additional_datamodule_builders=[from_toxic],
        default_arguments={
            "trainer.max_epochs": 3,
        },
        datamodule_attributes={"num_classes", "multi_label", "backbone"},
    )

    cli.trainer.save_checkpoint("text_classification_model.pt")


if __name__ == "__main__":
    text_classification()
