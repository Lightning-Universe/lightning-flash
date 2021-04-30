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
        if str(keys) == keys:
            keys = [keys]
        self.keys = keys

    def forward(self, x: Mapping[str, Any]) -> Mapping[str, Any]:
        inputs = [x[key] for key in filter(lambda key: key in x, self.keys)]
        if len(inputs) > 0:
            outputs = super().forward(*inputs)
            if not isinstance(outputs, tuple):
                outputs = (outputs, )

            result = {}
            result.update(x)
            for i, key in enumerate(self.keys):
                result[key] = outputs[i]
            return result
        return x
