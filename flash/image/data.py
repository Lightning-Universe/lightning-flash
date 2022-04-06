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
from typing import Any, Dict, List

import numpy as np
import torch

import flash
from flash.core.data.io.input import DataKeys, Input, ServeInput
from flash.core.data.utilities.paths import filter_valid_files, has_file_allowed_extension, PATH_TYPE
from flash.core.data.utilities.samples import to_samples
from flash.core.data.utils import image_default_loader
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE, Image, requires

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets.folder import IMG_EXTENSIONS
    from torchvision.transforms.functional import to_pil_image
else:
    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

NP_EXTENSIONS = (".npy",)


def image_loader(filepath: str):
    if has_file_allowed_extension(filepath, IMG_EXTENSIONS):
        img = image_default_loader(filepath)
    elif has_file_allowed_extension(filepath, NP_EXTENSIONS):
        img = Image.fromarray(np.load(filepath).astype("uint8"), "RGB")
    else:
        raise ValueError(
            f"File: {filepath} has an unsupported extension. Supported extensions: "
            f"{list(IMG_EXTENSIONS + NP_EXTENSIONS)}."
        )
    return img


class ImageDeserializer(ServeInput):
    @requires("image")
    def serve_load_sample(self, data: str) -> Dict:
        encoded_with_padding = (data + "===").encode("ascii")
        img = base64.b64decode(encoded_with_padding)
        buffer = BytesIO(img)
        img = Image.open(buffer, mode="r")
        return {
            DataKeys.INPUT: img,
        }

    @property
    def example_input(self) -> str:
        with (Path(flash.ASSETS_ROOT) / "fish.jpg").open("rb") as f:
            return base64.b64encode(f.read()).decode("UTF-8")


class ImageInput(Input):
    @requires("image")
    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        w, h = sample[DataKeys.INPUT].size  # W x H
        if DataKeys.METADATA not in sample:
            sample[DataKeys.METADATA] = {}
        sample[DataKeys.METADATA]["size"] = (h, w)
        return sample


class ImageFilesInput(ImageInput):
    def load_data(self, files: List[PATH_TYPE]) -> List[Dict[str, Any]]:
        files = filter_valid_files(files, valid_extensions=IMG_EXTENSIONS + NP_EXTENSIONS)
        return to_samples(files)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        filepath = sample[DataKeys.INPUT]
        sample[DataKeys.INPUT] = image_loader(filepath)
        sample = super().load_sample(sample)
        sample[DataKeys.METADATA]["filepath"] = filepath
        return sample


class ImageTensorInput(ImageInput):
    def load_data(self, tensor: Any) -> List[Dict[str, Any]]:
        return to_samples(tensor)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        img = to_pil_image(sample[DataKeys.INPUT])
        sample[DataKeys.INPUT] = img
        return super().load_sample(sample)


class ImageNumpyInput(ImageInput):
    def load_data(self, array: Any) -> List[Dict[str, Any]]:
        return to_samples(array)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        img = to_pil_image(torch.from_numpy(sample[DataKeys.INPUT]))
        sample[DataKeys.INPUT] = img
        return super().load_sample(sample)
