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
import os
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch_geometric.data import DataLoader, Dataset

from flash.data.base_viz import BaseVisualization  # for viz
from flash.data.callback import BaseDataFetcher
from flash.data.data_module import DataModule
from flash.data.data_source import DefaultDataKeys, DefaultDataSources
from flash.data.process import Preprocess
from torch_geometric.transforms import default_transforms, train_default_transforms
from flash.graph.data import GraphPathsDataSource
from flash.utils.imports import _MATPLOTLIB_AVAILABLE

# See https://1176-333857397-gh.circle-artifacts.com/0/html/task_template.html

class GraphClassificationPreprocess(Preprocess):

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        num_features: int = 128
    ):
        self.num_features = num_features #todo: probably wrong

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.FILES: GraphPathsDataSource(),
                DefaultDataSources.FOLDERS: GraphPathsDataSource()
            },
            default_data_source=DefaultDataSources.FILES,
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {**self.transforms, "num_features": self.num_features}

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(**state_dict)
    # Seems like there are no default. Also importantly transforms are called on DataSet.
    # For example see https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/dataset.html#Dataset.__getitem__
    #todo: do we want to implement transforms here?
    
    '''
    def default_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_transforms()

    def train_default_transforms(self) -> Optional[Dict[str, Callable]]:
        return train_default_transforms()
    '''


class GraphClassificationData(DataModule):
    """Data module for graph classification tasks."""

    preprocess_cls = GraphClassificationPreprocess