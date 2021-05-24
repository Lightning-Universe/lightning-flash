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

from torch.utils.data import Dataset

from flash.core.data.auto_dataset import AutoDataset
from flash.core.data.data_source import DatasetDataSource, DefaultDataKeys, PathsDataSource
from flash.core.utilities.imports import _PYTORCH_GEOMETRIC_AVAILABLE

if _PYTORCH_GEOMETRIC_AVAILABLE:
    from torch_geometric.data import Dataset as PyGDataset

class GraphDatasetSource(DatasetDataSource):

    def load_data(self, dataset: Dataset, auto_dataset: AutoDataset) -> Dataset:
        data = super().load_data(dataset, auto_dataset)
        if self.training:
            if isinstance(dataset, PyGDataset):
                auto_dataset.num_classes = dataset.num_classes
                auto_dataset.num_features = dataset.num_features
        return data