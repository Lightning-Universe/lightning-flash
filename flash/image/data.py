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

import numpy as np
import torch

import flash
from flash.core.data.data_source import (
    DefaultDataKeys,
    FiftyOneDataSource,
    has_file_allowed_extension,
    NumpyDataSource,
    PathsDataSource,
    TensorDataSource,
)
from flash.core.data.process import Deserializer
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE, Image, requires

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
    from torchvision.transforms.functional import to_pil_image
else:
    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


NP_EXTENSIONS = (".npy",)


def image_loader(filepath: str):
    if has_file_allowed_extension(filepath, IMG_EXTENSIONS):
        img = default_loader(filepath)
    elif has_file_allowed_extension(filepath, NP_EXTENSIONS):
        img = Image.fromarray(np.load(filepath).astype("uint8"), "RGB")
    else:
        raise ValueError(
            f"File: {filepath} has an unsupported extension. Supported extensions: "
            f"{list(IMG_EXTENSIONS + NP_EXTENSIONS)}."
        )
    return img


class ImageDeserializer(Deserializer):
    @requires("image")
    def deserialize(self, data: str) -> Dict:
        encoded_with_padding = (data + "===").encode("ascii")
        img = base64.b64decode(encoded_with_padding)
        buffer = BytesIO(img)
        img = Image.open(buffer, mode="r")
        return {
            DefaultDataKeys.INPUT: img,
        }

    @property
    def example_input(self) -> str:
        with (Path(flash.ASSETS_ROOT) / "fish.jpg").open("rb") as f:
            return base64.b64encode(f.read()).decode("UTF-8")


class ImagePathsDataSource(PathsDataSource):
    def __init__(self):
        super().__init__(loader=image_loader, extensions=IMG_EXTENSIONS + NP_EXTENSIONS)

    @requires("image")
    def load_sample(self, sample: Dict[str, Any], dataset: Optional[Any] = None) -> Dict[str, Any]:
        sample = super().load_sample(sample, dataset)
        w, h = sample[DefaultDataKeys.INPUT].size  # WxH
        sample[DefaultDataKeys.METADATA]["size"] = (h, w)
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
    @staticmethod
    def load_sample(sample: Dict[str, Any], dataset: Optional[Any] = None) -> Dict[str, Any]:
        img_path = sample[DefaultDataKeys.INPUT]
        img = default_loader(img_path)
        sample[DefaultDataKeys.INPUT] = img
        w, h = img.size  # WxH
        sample[DefaultDataKeys.METADATA] = {
            "filepath": img_path,
            "size": (h, w),
        }
        return sample
