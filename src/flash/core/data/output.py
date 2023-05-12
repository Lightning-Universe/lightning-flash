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
from typing import Any, List, Union

from flash.core.data.io.input import DataKeys
from flash.core.data.io.output import Output
from flash.core.registry import FlashRegistry

BASE_OUTPUTS = FlashRegistry("outputs")
BASE_OUTPUTS(name="raw")(Output)


@BASE_OUTPUTS(name="preds")
class PredsOutput(Output):
    """A :class:`~flash.core.data.io.output.Output` which returns the "preds" from the model outputs."""

    def transform(self, sample: Any) -> Union[int, List[int]]:
        return sample.get(DataKeys.PREDS, sample) if isinstance(sample, dict) else sample
