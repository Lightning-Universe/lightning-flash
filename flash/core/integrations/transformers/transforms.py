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
from typing import Any, Callable, Dict

import torch

from flash.core.data.input_transform import InputTransform


def to_tensor(sample: Dict[str, Any]) -> Dict[str, Any]:
    for key in sample:
        sample[key] = torch.as_tensor(sample[key])
    return sample


@dataclass
class TransformersInputTransform(InputTransform):
    def per_sample_transform(self) -> Callable:
        return to_tensor
