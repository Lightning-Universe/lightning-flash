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
from flash.tabular.classification.data import TabularClassificationData
from flash.tabular.classification.model import TabularClassifier

__all__ = ["tabular_classification"]


def from_titanic(
    val_split: float = 0.1,
    batch_size: int = 4,
    **data_module_kwargs,
) -> TabularClassificationData:
    """Downloads and loads the Titanic data set."""
    download_data("https://pl-flash-data.s3.amazonaws.com/titanic.zip", "./data")
    return TabularClassificationData.from_csv(
        ["Sex", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"],
        "Fare",
        target_fields="Survived",
        train_file="data/titanic/titanic.csv",
        val_split=val_split,
        batch_size=batch_size,
        **data_module_kwargs,
    )


def tabular_classification():
    """Classify tabular data."""
    cli = FlashCLI(
        TabularClassifier,
        TabularClassificationData,
        default_datamodule_builder=from_titanic,
        default_arguments={
            "trainer.max_epochs": 3,
            "model.backbone": "tabnet",
        },
        finetune=False,
        datamodule_attributes={
            "parameters",
            "embedding_sizes",
            "cat_dims",
            "num_features",
            "num_classes",
            "labels",
        },
    )

    cli.trainer.save_checkpoint("tabular_classification_model.pt")


if __name__ == "__main__":
    tabular_classification()
