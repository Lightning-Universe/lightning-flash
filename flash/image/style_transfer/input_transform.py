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
from dataclasses import dataclass
from typing import Callable

from flash.core.data.io.input import DataKeys
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.transforms import ApplyToKeys
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as T


@dataclass
class StyleTransferInputTransform(InputTransform):

    image_size: int = 256

    def per_sample_transform(self) -> Callable:
        return ApplyToKeys(
            DataKeys.INPUT, T.Compose([T.ToTensor(), T.Resize(self.image_size), T.CenterCrop(self.image_size)])
        )
