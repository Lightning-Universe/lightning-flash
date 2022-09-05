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
from typing import Any, Mapping, Sequence, Union

import numpy as np
from torch import nn

from flash.core.data.io.input import DataKeys
from flash.core.data.utils import convert_to_modules
from flash.core.utilities.imports import _ALBUMENTATIONS_AVAILABLE, requires

if _ALBUMENTATIONS_AVAILABLE:
    from albumentations import BasicTransform, Compose
else:
    BasicTransform, Compose = object, object


class AlbumentationsAdapter(nn.Module):
    # mapping from albumentations to Flash
    TRANSFORM_INPUT_MAPPING = {"image": DataKeys.INPUT, "mask": DataKeys.TARGET}

    @requires("albumentations")
    def __init__(
        self,
        transform: Union[BasicTransform, Sequence[BasicTransform]],
        mapping: dict = None,
    ):
        super().__init__()
        if not isinstance(transform, (list, tuple)):
            transform = [transform]
        self.transform = Compose(list(transform))
        if not mapping:
            mapping = self.TRANSFORM_INPUT_MAPPING
        self._mapping_rev = mapping
        self._mapping = {v: k for k, v in mapping.items()}

    def forward(self, x: Any) -> Any:
        if isinstance(x, dict):
            x_ = {self._mapping.get(key, key): np.array(value) for key, value in x.items() if key in self._mapping}
        else:
            x_ = {"image": x}
        x_ = self.transform(**x_)
        if isinstance(x, dict):
            x.update({self._mapping_rev.get(k, k): x_[k] for k in self._mapping_rev if k in x_})
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
                raise Exception("Failed to apply transforms to multiple keys at the same time.") from e

            for i, key in enumerate(keys):
                result[key] = outputs[i]

        # result is simply returned if len(inputs) == 0
        return result

    def __repr__(self):
        transform = list(self.children())

        keys = self.keys[0] if len(self.keys) == 1 else self.keys
        transform = transform[0] if len(transform) == 1 else transform

        return f"{self.__class__.__name__}(keys={repr(keys)}, transform={repr(transform)})"
