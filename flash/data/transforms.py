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

from torch import nn

from flash.data.utils import convert_to_modules


class ApplyToKeys(nn.Sequential):

    def __init__(self, keys: Union[str, Sequence[str]], *args):
        super().__init__(*[convert_to_modules(arg) for arg in args])
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys

    def forward(self, x: Mapping[str, Any]) -> Mapping[str, Any]:
        keys = list(filter(lambda key: key in x, self.keys))
        inputs = [x[key] for key in keys]
        if len(inputs) > 0:
            if len(inputs) == 1:
                inputs = inputs[0]
            outputs = super().forward(inputs)
            if not isinstance(outputs, Sequence):
                outputs = (outputs, )

            result = {}
            result.update(x)
            for i, key in enumerate(keys):
                result[key] = outputs[i]
            return result
        return x
