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
import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import torch

import flash
from flash.core.data.data_source import (
    DefaultDataKeys,
    FiftyOneDataSource,
    NumpyDataSource,
    PathsDataSource,
    TensorDataSource,
)
from flash.core.data.process import Deserializer
from flash.core.utilities.imports import _PIL_AVAILABLE, _requires_extras, _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    import torchvision
    from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
    from torchvision.transforms.functional import to_pil_image
else:
    IMG_EXTENSIONS = []

if _PIL_AVAILABLE:
    from PIL import Image as PILImage
else:

    class Image:
        Image = None


class ImageDeserializer(Deserializer):

    @_requires_extras("image")
    def __init__(self):
        super().__init__()
        self.to_tensor = torchvision.transforms.ToTensor()

    def deserialize(self, data: str) -> Dict:
        encoded_with_padding = (data + "===").encode("ascii")
        img = base64.b64decode(encoded_with_padding)
        buffer = BytesIO(img)
        img = PILImage.open(buffer, mode="r")
        return {
            DefaultDataKeys.INPUT: img,
        }

    @property
    def example_input(self) -> str:
        with (Path(flash.ASSETS_ROOT) / "fish.jpg").open("rb") as f:
            return base64.b64encode(f.read()).decode("UTF-8")


class ImagePathsDataSource(PathsDataSource):

    @_requires_extras("image")
    def __init__(self):
        super().__init__(extensions=IMG_EXTENSIONS)

    def load_sample(self, sample: Dict[str, Any], dataset: Optional[Any] = None) -> Dict[str, Any]:
        img_path = sample[DefaultDataKeys.INPUT]
        img = default_loader(img_path)
        sample[DefaultDataKeys.INPUT] = img
        w, h = img.size  # WxH
        sample[DefaultDataKeys.METADATA] = {
            "filepath": img_path,
            "size": (h, w),
        }
        return sample


class ImageTensorDataSource(TensorDataSource):

    def load_sample(self, sample: Dict[str, Any], dataset: Optional[Any] = None) -> Dict[str, Any]:
        img = to_pil_image(sample[DefaultDataKeys.INPUT])
        sample[DefaultDataKeys.INPUT] = img
        w, h = img.size  # WxH
        sample[DefaultDataKeys.METADATA] = {"size": (h, w)}
        return sample


class ImageNumpyDataSource(NumpyDataSource):

    def load_sample(self, sample: Dict[str, Any], dataset: Optional[Any] = None) -> Dict[str, Any]:
        img = to_pil_image(torch.from_numpy(sample[DefaultDataKeys.INPUT]))
        sample[DefaultDataKeys.INPUT] = img
        w, h = img.size  # WxH
        sample[DefaultDataKeys.METADATA] = {"size": (h, w)}
        return sample


class ImageFiftyOneDataSource(FiftyOneDataSource):

    def load_sample(self, sample: Dict[str, Any], dataset: Optional[Any] = None) -> Dict[str, Any]:
        img_path = sample[DefaultDataKeys.INPUT]
        img = default_loader(img_path)
        sample[DefaultDataKeys.INPUT] = img
        w, h = img.size  # WxH
        sample[DefaultDataKeys.METADATA] = {
            "filepath": img_path,
            "size": (h, w),
        }
        return sample
