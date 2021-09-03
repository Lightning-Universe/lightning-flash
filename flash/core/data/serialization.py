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

from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.process import Serializer


class Preds(Serializer):
    """A :class:`~flash.core.data.process.Serializer` which returns the "preds" from the model outputs."""

    def serialize(self, sample: Any) -> Union[int, List[int]]:
        return sample.get(DefaultDataKeys.PREDS, sample) if isinstance(sample, dict) else sample
