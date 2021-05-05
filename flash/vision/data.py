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

import torch
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torchvision.transforms.functional import to_pil_image

from flash.data.data_source import (
    DefaultDataKeys,
    FilesDataSource,
    FoldersDataSource,
    NumpyDataSource,
    TensorDataSource,
)


class ImageFoldersDataSource(FoldersDataSource):

    def __init__(self):
        super().__init__(extensions=IMG_EXTENSIONS)

    def load_sample(self, sample: Mapping[str, Any], dataset: Optional[Any] = None) -> Any:
        result = {}  # TODO: this is required to avoid a memory leak, can we automate this?
        result.update(sample)
        result[DefaultDataKeys.INPUT] = default_loader(sample[DefaultDataKeys.INPUT])
        return result


class ImageFilesDataSource(FilesDataSource):

    def __init__(self):
        super().__init__(extensions=IMG_EXTENSIONS)

    def load_sample(self, sample: Mapping[str, Any], dataset: Optional[Any] = None) -> Any:
        result = {}  # TODO: this is required to avoid a memory leak, can we automate this?
        result.update(sample)
        result[DefaultDataKeys.INPUT] = default_loader(sample[DefaultDataKeys.INPUT])
        return result


class ImageTensorDataSource(TensorDataSource):

    def load_sample(self, sample: Dict[str, Any], dataset: Optional[Any] = None) -> Any:
        sample[DefaultDataKeys.INPUT] = to_pil_image(sample[DefaultDataKeys.INPUT])
        return sample


class ImageNumpyDataSource(NumpyDataSource):

    def load_sample(self, sample: Dict[str, Any], dataset: Optional[Any] = None) -> Any:
        sample[DefaultDataKeys.INPUT] = to_pil_image(torch.from_numpy(sample[DefaultDataKeys.INPUT]))
        return sample
