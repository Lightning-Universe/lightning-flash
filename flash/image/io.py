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
from typing import Any, Dict

import numpy as np
import torch
from pytorch_lightning.trainer.states import RunningStage

from flash.core.data_v2.io.input import (
    FiftyOneInput,
    has_file_allowed_extension,
    INPUT_TRANSFORM_TYPE,
    InputDataKeys,
    NumpyInput,
    PathsInput,
    TensorInput,
)
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


class ImagePathsInput(PathsInput):
    def __init__(
        self,
        running_stage: RunningStage,
        transform: INPUT_TRANSFORM_TYPE,
    ):
        super().__init__(
            running_stage=running_stage,
            transform=transform,
            loader=image_loader,
            extensions=IMG_EXTENSIONS + NP_EXTENSIONS,
        )

    @requires("image")
    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = super().load_sample(sample)
        w, h = sample[InputDataKeys.INPUT].size  # WxH
        sample[InputDataKeys.METADATA]["size"] = (h, w)
        return sample


class ImageTensorInput(TensorInput):
    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        img = to_pil_image(sample[InputDataKeys.INPUT])
        sample[InputDataKeys.INPUT] = img
        w, h = img.size  # WxH
        sample[InputDataKeys.METADATA] = {"size": (h, w)}
        return sample


class ImageNumpyInput(NumpyInput):
    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        img = to_pil_image(torch.from_numpy(sample[InputDataKeys.INPUT]))
        sample[InputDataKeys.INPUT] = img
        w, h = img.size  # WxH
        sample[InputDataKeys.METADATA] = {"size": (h, w)}
        return sample


class ImageFiftyOneInput(FiftyOneInput):
    @staticmethod
    def load_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
        img_path = sample[InputDataKeys.INPUT]
        img = default_loader(img_path)
        sample[InputDataKeys.INPUT] = img
        w, h = img.size  # WxH
        sample[InputDataKeys.METADATA] = {
            "filepath": img_path,
            "size": (h, w),
        }
        return sample
