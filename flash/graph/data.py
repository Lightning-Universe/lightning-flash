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
from typing import Any, Dict, Optional

import torch

from flash.data.data_source import DefaultDataKeys, NumpyDataSource, PathsDataSource, TensorDataSource

#todo: how to get default_loader and GRAPH_EXTENSIONS? These were provided by torchvision in the case of vision

class GraphPathsDataSource(PathsDataSource):

    def __init__(self):
        super().__init__(extensions=GRAPH_EXTENSIONS)

    def load_sample(self, sample: Dict[str, Any], dataset: Optional[Any] = None) -> Dict[str, Any]:
        sample[DefaultDataKeys.INPUT] = default_loader(sample[DefaultDataKeys.INPUT])
        return sample


class ImageTensorDataSource(TensorDataSource):

    def load_sample(self, sample: Dict[str, Any], dataset: Optional[Any] = None) -> Dict[str, Any]:
        #todo
        #Example from vision: sample[DefaultDataKeys.INPUT] = to_pil_image(sample[DefaultDataKeys.INPUT])
        return sample


class ImageNumpyDataSource(NumpyDataSource): #todo: is this needed?

    def load_sample(self, sample: Dict[str, Any], dataset: Optional[Any] = None) -> Dict[str, Any]:
        #todo
        #Example from vision: sample[DefaultDataKeys.INPUT] = to_pil_image(torch.from_numpy(sample[DefaultDataKeys.INPUT]))
        return sample
