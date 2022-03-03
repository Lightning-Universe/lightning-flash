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
from typing import Any, Dict, Mapping, Optional

from torch.utils.data import Dataset

from flash.core.data.io.classification_input import ClassificationInputMixin
from flash.core.data.io.input import DataKeys, Input
from flash.core.data.utilities.classification import TargetFormatter
from flash.core.data.utilities.samples import to_sample
from flash.core.utilities.imports import _GRAPH_AVAILABLE, requires

if _GRAPH_AVAILABLE:
    from torch_geometric.data import Data, InMemoryDataset


def _get_num_features(sample: Dict[str, Any]) -> Optional[int]:
    """Get the number of features per node in the given dataset."""
    data = sample[DataKeys.INPUT]
    data = data[0] if isinstance(data, tuple) else data
    return getattr(data, "num_node_features", None)


class GraphClassificationDatasetInput(Input, ClassificationInputMixin):
    num_features: int
    num_classes: int

    @requires("graph")
    def load_data(self, dataset: Dataset, target_formatter: Optional[TargetFormatter] = None) -> Dataset:
        if not self.predicting:
            self.num_features = _get_num_features(self.load_sample(dataset[0]))

            if isinstance(dataset, InMemoryDataset):
                self.load_target_metadata([sample.y for sample in dataset], target_formatter)
            else:
                self.load_target_metadata(None, target_formatter)

            if hasattr(dataset, "num_classes"):
                self.num_classes = dataset.num_classes
        return dataset

    def load_sample(self, sample: Any) -> Mapping[str, Any]:
        if isinstance(sample, Data):
            sample = (sample, sample.y)
        sample = to_sample(sample)
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = self.format_target(sample[DataKeys.TARGET])
        return sample
