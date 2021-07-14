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
from typing import Any

from torch.utils.data import Dataset

from flash.core.data.data_source import DatasetDataSource
from flash.core.utilities.imports import _TORCH_GEOMETRIC_AVAILABLE, requires_extras

if _TORCH_GEOMETRIC_AVAILABLE:
    from torch_geometric.data import Dataset as PyGDataset


class GraphDatasetDataSource(DatasetDataSource):

    @requires_extras("graph")
    def load_data(self, data: Dataset, dataset: Any = None) -> Dataset:
        data = super().load_data(data, dataset)
        if self.training:
            if isinstance(data, PyGDataset):
                dataset.num_classes = data.num_classes
                dataset.num_features = data.num_features
        return data
