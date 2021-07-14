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

from typing import Sequence, Tuple, Union
import copy

from torch.utils.data import Dataset

from flash.core.data.auto_dataset import AutoDataset
from flash.core.data.data_source import DatasetDataSource, DefaultDataKeys, SequenceDataSource
from flash.core.utilities.imports import _PYTORCH_GEOMETRIC_AVAILABLE

if _PYTORCH_GEOMETRIC_AVAILABLE:
    from torch_geometric.data import Data as PyGData
    from torch_geometric.data import Dataset as PyGDataset


class GraphDatasetSource(DatasetDataSource):

    def load_data(self, dataset: Dataset, auto_dataset: AutoDataset) -> Dataset:
        data = super().load_data(dataset, auto_dataset)
        if self.training:
            if isinstance(dataset, PyGDataset):
                auto_dataset.num_classes = dataset.num_classes
                auto_dataset.num_features = dataset.num_features
        return data

class GraphSequenceDataSource(SequenceDataSource):

    def load_data(self, data_list: Sequence[PyGData]) -> Sequence:
        # Converting the PyGDataList to the tuple of sequences that load_data expects:

        # Recover the labels
        data_list_y = [data_list[i].y for i in range(len(data_list))]

        # Recover the data
        data_list_x = copy(data_list)
        for data_list_i in data_list_x:
            data_list_i.y = None
        
        # Create data_list
        data_list = (data_list_x, data_list_y)
        data = super().load_data(data_list)

        return data
