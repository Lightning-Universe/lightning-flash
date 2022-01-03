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
from typing import Any, Mapping

from torch.utils.data import Dataset

from flash.core.data.data_module import DatasetInput
from flash.core.data.io.classification_input import ClassificationInput, ClassificationState
from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _GRAPH_AVAILABLE, requires

if _GRAPH_AVAILABLE:
    from torch_geometric.data import Data
    from torch_geometric.data import Dataset as TorchGeometricDataset
    from torch_geometric.data import InMemoryDataset


class GraphClassificationDatasetInput(DatasetInput, ClassificationInput):
    @requires("graph")
    def load_data(self, dataset: Dataset) -> Dataset:
        if not self.predicting:
            if isinstance(dataset, TorchGeometricDataset):
                self.num_features = dataset.num_features

                if isinstance(dataset, InMemoryDataset):
                    self.load_target_metadata([sample.y for sample in dataset])
                else:
                    self.num_classes = dataset.num_classes
                    self.set_state(ClassificationState(self.labels, self.num_classes))
        return dataset

    def load_sample(self, sample: Any) -> Mapping[str, Any]:
        if isinstance(sample, Data):
            sample = {DataKeys.INPUT: sample, DataKeys.TARGET: sample.y}
            if getattr(self, "target_mode", None) is not None:
                sample[DataKeys.TARGET] = self.format_target(sample[DataKeys.TARGET])
            return sample
        return super().load_sample(sample)
