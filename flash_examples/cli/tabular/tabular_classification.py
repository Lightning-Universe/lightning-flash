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
from flash.tabular import TabularClassificationData, TabularClassifier  # noqa: E402


def from_titanic(
    batch_size: int = 4,
    num_workers: Optional[int] = None,
    **preprocess_kwargs,
) -> TabularClassificationData:
    """Downloads and loads the Titanic data set."""
    download_data("https://pl-flash-data.s3.amazonaws.com/titanic.zip", "./data")
    return TabularClassificationData.from_csv(
        ["Sex", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"],
        "Fare",
        target_fields="Survived",
        train_file="data/titanic/titanic.csv",
        val_split=0.1,
        batch_size=batch_size,
        num_workers=num_workers,
        **preprocess_kwargs,
    )


# 1. Build the model, datamodule, and trainer. Expose them through CLI. Fine-tune
cli = FlashCLI(
    TabularClassifier,
    TabularClassificationData,
    default_datamodule_builder=from_titanic,
    default_arguments={
        "trainer.max_epochs": 3,
    },
    finetune=False,
    datamodule_attributes={"num_features", "num_classes", "embedding_sizes"},
)

# 2. Save the model!
cli.trainer.save_checkpoint("tabular_classification_model.pt")
