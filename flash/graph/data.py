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

from flash.core.data.io.input_base import Input
from flash.core.utilities.imports import _GRAPH_AVAILABLE, requires

if _GRAPH_AVAILABLE:
    from torch_geometric.data import Data
    from torch_geometric.data import Dataset as TorchGeometricDataset


class GraphDatasetInput(Input):
    @requires("graph")
    def load_data(self, dataset: Dataset) -> Dataset:
        if not self.predicting:
            if isinstance(dataset, TorchGeometricDataset):
                self.num_classes = dataset.num_classes
                self.num_features = dataset.num_features
        return dataset

    def load_sample(self, sample: Any) -> Mapping[str, Any]:
        if isinstance(sample, Data):
            return sample
        return super().load_sample(sample)
