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
from typing import Any, Callable, Dict, Optional, Sequence

from flash.core.data.data_module import DataModule
from flash.core.data.data_source import DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Preprocess
from flash.core.data.transforms import ApplyToKeys
from flash.core.utilities.imports import _PYTORCH_GEOMETRIC_AVAILABLE
from flash.graph.data_source import GraphDatasetSource, GraphSequenceDataSource

if _PYTORCH_GEOMETRIC_AVAILABLE:
    import networkx as nx
    from torch_geometric.data import DataLoader, Dataset
    from torch_geometric.data.batch import Batch

# See https://1176-333857397-gh.circle-artifacts.com/0/html/task_template.html


class GraphClassificationPreprocess(Preprocess):

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        num_features: int = 128  #todo: do we want to add backbone here as in text?
    ):
        self.num_features = num_features
        if not _PYTORCH_GEOMETRIC_AVAILABLE:
            raise ModuleNotFoundError("Please, pip install -e '.[graph]'")

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.DATASET: GraphDatasetSource(),
                DefaultDataSources.SEQUENCE: GraphSequenceDataSource()
            },
            default_data_source=DefaultDataSources.DATASET,
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {**self.transforms, "num_features": self.num_features}

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(**state_dict)

    def collate(self, samples: Sequence) -> Any:
        return {DefaultDataKeys.INPUT: Batch.from_data_list([s[DefaultDataKeys.INPUT] for s in samples])}


class GraphClassificationData(DataModule):
    """Data module for graph classification tasks."""

    preprocess_cls = GraphClassificationPreprocess
