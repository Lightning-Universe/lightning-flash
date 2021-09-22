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
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader, Dataset

from flash import DataModule
from flash.core.data.auto_dataset import BaseAutoDataset
from flash.core.data.data_pipeline import DataPipeline
from flash.core.utilities.imports import _BAAL_AVAILABLE, requires

if _BAAL_AVAILABLE:
    from baal.active.dataset import ActiveLearningDataset
    from baal.active.heuristics import AbstractHeuristic, BALD
else:

    class AbstractHeuristic:
        pass

    class BALD(AbstractHeuristic):
        def __init__(self, reduction: Callable):
            super().__init__()


def dataset_to_non_labelled_tensor(dataset: BaseAutoDataset) -> torch.tensor:
    return torch.zeros(len(dataset))


def filter_unlabelled_data(dataset: BaseAutoDataset) -> Dataset:
    return dataset


class ActiveLearningDataModule(LightningDataModule):
    @requires("baal")
    def __init__(
        self,
        labelled: Optional[DataModule] = None,
        heuristic: "AbstractHeuristic" = BALD(reduction=np.mean),
        map_dataset_to_labelled: Optional[Callable] = dataset_to_non_labelled_tensor,
        filter_unlabelled_data: Optional[Callable] = filter_unlabelled_data,
    ):
        """The `ActiveLearningDataModule` handles data manipulation for ActiveLearning.

        Args:
            labelled: DataModule containing labelled train data for research use-case.
                The labelled data would be masked.
            heuristic: Sorting algorithm used to rank samples on how likely they can help with model performance.
            map_dataset_to_labelled: Function used to emulate masking on labelled dataset.
            filter_unlabelled_data: Function used to filter the unlabelled data while computing uncertainties.
        """
        self.labelled = labelled
        self.heuristic = heuristic
        self.map_dataset_to_labelled = map_dataset_to_labelled
        self.filter_unlabelled_data = filter_unlabelled_data
        self._dataset: Optional[ActiveLearningDataset] = None

        if not self.labelled:
            raise MisconfigurationException("The labelled `datamodule` should be provided.")

        if not self.labelled.num_classes:
            raise MisconfigurationException("The labelled dataset should be labelled")

        if self.labelled and (
            self.labelled._val_ds is not None
            or self.labelled._test_ds is not None
            or self.labelled._predict_ds is not None
        ):
            raise MisconfigurationException("The labelled `datamodule` should have only train data.")

        self._dataset = ActiveLearningDataset(
            self.labelled._train_ds, labelled=self.map_dataset_to_labelled(self.labelled._train_ds)
        )

    @property
    def has_labelled_data(self) -> bool:
        return self._dataset.n_labelled > 0

    @property
    def has_unlabelled_data(self) -> bool:
        return self._dataset.n_unlabelled > 0

    @property
    def num_classes(self) -> Optional[int]:
        return getattr(self.labelled, "num_classes", None) or getattr(self.unlabelled, "num_classes", None)

    @property
    def data_pipeline(self) -> "DataPipeline":
        return self.labelled.data_pipeline

    def train_dataloader(self) -> "DataLoader":
        self.labelled._train_ds = self._dataset
        return self.labelled.train_dataloader()

    def predict_dataloader(self) -> "DataLoader":
        self.labelled._train_ds = self.filter_unlabelled_data(self._dataset.pool)
        return self.labelled.train_dataloader()

    def label(self, predictions: Any = None, indices=None):
        if predictions and indices:
            raise MisconfigurationException(
                "The `predictions` and `indices` are mutually exclusive, pass only of one them."
            )
        if predictions:
            uncertainties = [self.heuristic.get_uncertainties(np.asarray(p)) for idx, p in enumerate(predictions)]
            indices = np.argsort(uncertainties)
        if self._dataset is not None:
            self._dataset.labelled[indices] = True

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self._dataset.state_dict()

    def load_state_dict(self, state_dict) -> None:
        return self._dataset.load_state_dict(state_dict)
