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
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException

import networkx as nx
from torch_geometric.data import Dataset, DataLoader


from flash.core.classification import ClassificationDataPipeline
from flash.core.data.datamodule import DataModule
from flash.core.data.utils import _contains_any_tensor

'''
The structure we follow is DataSet -> DataLoader -> DataModule -> DataPipeline
'''

class BasicDataset(Dataset):
    def __init__(self, root = None, processed_dir = 'processed', raw_dir = 'raw',  transform=None, pre_transform=None, pre_filter=None):

        super(BasicDataset, self).__init__(root, transform, pre_transform, pre_filter)

        self._processed_dir = processed_dir
        self._raw_dir = raw_dir

    @property
    def raw_dir(self):
        return os.path.join(self.root, self._raw_dir)

    @property
    def processed_dir(self):
        '''self.processed_dir already has root on it'''
        return os.path.join(self.root, self._processed_dir)

    @property
    def raw_file_names(self):
        '''self.raw_dir already has root on it'''
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return os.listdir(self.processed_dir)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        #TODO: Is data.pt the best way/file type to load the data? 
        #TODO: Interface with networkx would probably go here with some option to say how to load it
        return data