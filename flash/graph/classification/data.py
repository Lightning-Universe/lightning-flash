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

from flash.core.data.data_module import DataModule
from flash.core.data.data_source import DefaultDataSources
from flash.core.data.process import Preprocess
from flash.core.utilities.imports import _GRAPH_AVAILABLE
from flash.graph.data import GraphDatasetDataSource

if _GRAPH_AVAILABLE:
    from torch_geometric.data.batch import Batch
    from torch_geometric.transforms import NormalizeFeatures


class GraphClassificationPreprocess(Preprocess):
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
    ):
        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.DATASETS: GraphDatasetDataSource(),
            },
            default_data_source=DefaultDataSources.DATASETS,
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return self.transforms

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(**state_dict)

    @staticmethod
    def default_transforms() -> Optional[Dict[str, Callable]]:
        return {"pre_tensor_transform": NormalizeFeatures(), "collate": Batch.from_data_list}


class GraphClassificationData(DataModule):
    """Data module for graph classification tasks."""

    preprocess_cls = GraphClassificationPreprocess

    @property
    def num_features(self):
        n_cls_train = getattr(self.train_dataset, "num_features", None)
        n_cls_val = getattr(self.val_dataset, "num_features", None)
        n_cls_test = getattr(self.test_dataset, "num_features", None)
        return n_cls_train or n_cls_val or n_cls_test
