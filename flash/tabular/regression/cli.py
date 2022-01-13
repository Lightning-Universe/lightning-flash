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
from flash.tabular.regression.data import TabularRegressionData
from flash.tabular.regression.model import TabularRegressor

__all__ = ["tabular_regression"]


def from_titanic(
    val_split: float = 0.1,
    batch_size: int = 4,
    **data_module_kwargs,
) -> TabularRegressionData:
    """Downloads and loads the Titanic data set."""
    download_data("https://pl-flash-data.s3.amazonaws.com/titanic.zip", "./data")
    return TabularRegressionData.from_csv(
        ["Sex", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"],
        None,
        target_field="Fare",
        train_file="data/titanic/titanic.csv",
        val_split=val_split,
        batch_size=batch_size,
        **data_module_kwargs,
    )


def tabular_regression():
    """Classify tabular data."""
    cli = FlashCLI(
        TabularRegressor,
        TabularRegressionData,
        default_datamodule_builder=from_titanic,
        default_arguments={
            "trainer.max_epochs": 3,
            "model.backbone": "tabnet",
        },
        finetune=False,
        datamodule_attributes={
            "embedding_sizes",
            "categorical_fields",
            "num_features",
            "cat_dims",
        },
    )

    cli.trainer.save_checkpoint("tabular_regression_model.pt")


if __name__ == "__main__":
    tabular_regression()
