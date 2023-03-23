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
from flash.core.utilities.flash_cli import FlashCLI
from flash.graph import GraphClassificationData, GraphClassifier

__all__ = ["graph_classification"]


def from_tu_dataset(
    name: str = "KKI",
    val_split: float = 0.1,
    batch_size: int = 4,
    **data_module_kwargs,
) -> GraphClassificationData:
    """Downloads and loads the TU Dataset."""
    from flash.core.utilities.imports import _TORCH_GEOMETRIC_AVAILABLE

    if _TORCH_GEOMETRIC_AVAILABLE:
        from torch_geometric.datasets import TUDataset
    else:
        raise ModuleNotFoundError("Please, pip install -e '.[graph]'")

    dataset = TUDataset(root="data", name=name)

    return GraphClassificationData.from_datasets(
        train_dataset=dataset,
        val_split=val_split,
        batch_size=batch_size,
        **data_module_kwargs,
    )


def graph_classification():
    """Classify graphs."""
    cli = FlashCLI(
        GraphClassifier,
        GraphClassificationData,
        default_datamodule_builder=from_tu_dataset,
        default_arguments={
            "trainer.max_epochs": 3,
        },
        finetune=False,
        datamodule_attributes={"num_classes", "labels", "num_features"},
    )

    cli.trainer.save_checkpoint("graph_classification.pt")


if __name__ == "__main__":
    graph_classification()
