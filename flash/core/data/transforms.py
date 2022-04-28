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
from typing import Any, Dict, Mapping, Sequence, Union

import numpy as np
import torch
from torch import nn, Tensor

from flash.core.data.io.input import DataKeys
from flash.core.data.utilities.collate import default_collate
from flash.core.data.utils import convert_to_modules
from flash.core.utilities.imports import _ALBUMENTATIONS_AVAILABLE, requires

if _ALBUMENTATIONS_AVAILABLE:
    from albumentations import BasicTransform, Compose
    from albumentations.pytorch import ToTensorV2
else:
    BasicTransform, Compose, ToTensorV2 = object, object, object


class AlbumentationsAdapter(nn.Module):
    # mapping from albumentations to Flash
    TRANSFORM_INPUT_MAPPING = {"image": DataKeys.INPUT, "mask": DataKeys.TARGET}

    @requires("albumentations")
    def __init__(
        self,
        transform: Union[BasicTransform, Sequence[BasicTransform]],
        mapping: dict = None,
        image_key: str = DataKeys.INPUT,
    ):
        super().__init__()
        if not isinstance(transform, (list, tuple)):
            transform = [transform]
        self.transform = Compose(list(transform) + [ToTensorV2()])
        self._img_key = image_key
        if not mapping:
            mapping = self.TRANSFORM_INPUT_MAPPING
        self._mapping_rev = mapping
        self._mapping = {v: k for k, v in mapping.items()}

    def _image_transform(self, x: Tensor) -> np.ndarray:
        if x.ndim == 3 and x.shape[0] < 4:
            return x.permute(1, 2, 0).numpy()
        return x.numpy()

    def forward(self, x: Any) -> Any:
        if isinstance(x, dict):
            x_ = {self._mapping.get(k, k): x[k].numpy() for k in self._mapping if k != self._img_key}
            if self._img_key in self._mapping:
                x_.update({self._mapping[self._img_key]: self._image_transform(x[self._img_key])})
        else:
            x_ = {"image": self._image_transform(x)}
        x_ = self.transform(**x_)
        if isinstance(x, dict):
            x.update({self._mapping_rev.get(k, k): x_[k] for k in self._mapping_rev})
        else:
            x = x_["image"]
        return x


class ApplyToKeys(nn.Sequential):
    """The ``ApplyToKeys`` class is an ``nn.Sequential`` which applies the given transforms to the given keys from
    the input. When a single key is given, a single value will be passed to the transforms. When multiple keys are
    given, the corresponding values will be passed to the transforms as a list.

    Args:
        keys: The key (``str``) or sequence of keys (``Sequence[str]``) to extract and forward to the transforms.
        args: The transforms, passed to the ``nn.Sequential`` super constructor.
    """

    def __init__(self, keys: Union[str, Sequence[str]], *args):
        super().__init__(*(convert_to_modules(arg) for arg in args))
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys

    def forward(self, x: Mapping[str, Any]) -> Mapping[str, Any]:
        keys = list(filter(lambda key: key in x, self.keys))
        inputs = [x[key] for key in keys]

        result = {}
        result.update(x)

        if len(inputs) == 1:
            result[keys[0]] = super().forward(inputs[0])
        elif len(inputs) > 1:
            try:
                outputs = super().forward(inputs)
            except TypeError as e:
                raise Exception(
                    "Failed to apply transforms to multiple keys at the same time,"
                    " try using KorniaParallelTransforms."
                ) from e

            for i, key in enumerate(keys):
                result[key] = outputs[i]

        # result is simply returned if len(inputs) == 0
        return result

    def __repr__(self):
        transform = list(self.children())

        keys = self.keys[0] if len(self.keys) == 1 else self.keys
        transform = transform[0] if len(transform) == 1 else transform

        return f"{self.__class__.__name__}(keys={repr(keys)}, transform={repr(transform)})"


class KorniaParallelTransforms(nn.Sequential):
    """The ``KorniaParallelTransforms`` class is an ``nn.Sequential`` which will apply the given transforms to each
    input (to ``.forward``) in parallel, whilst sharing the random state (``._params``). This should be used when
    multiple elements need to be augmented in the same way (e.g. an image and corresponding segmentation mask).

    Args:
        args: The transforms, passed to the ``nn.Sequential`` super constructor.
    """

    def __init__(self, *args):
        super().__init__(*(convert_to_modules(arg) for arg in args))

    def forward(self, inputs: Any):
        result = list(inputs) if isinstance(inputs, Sequence) else [inputs]
        for transform in self.children():
            inputs = result

            # we enforce the first time to sample random params
            result[0] = transform(inputs[0])

            if hasattr(transform, "_params") and bool(transform._params):
                params = transform._params
            else:
                params = None

            # apply transforms from (1, n)
            for i, input in enumerate(inputs[1:]):
                if params is not None:
                    result[i + 1] = transform(input, params)
                else:  # case for non-random transforms
                    result[i + 1] = transform(input)
            if hasattr(transform, "_params") and bool(transform._params):
                transform._params = None
        return result


def kornia_collate(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Kornia transforms add batch dimension which need to be removed.

    This function removes that dimension and then
    applies ``torch.utils.data._utils.collate.default_collate``.
    """
    if len(samples) == 1 and isinstance(samples[0], list):
        samples = samples[0]
    for sample in samples:
        for key in sample.keys():
            if torch.is_tensor(sample[key]) and sample[key].ndim == 4:
                sample[key] = sample[key].squeeze(0)
    return default_collate(samples)
