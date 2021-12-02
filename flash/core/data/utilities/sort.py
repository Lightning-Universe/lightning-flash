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
import re
from typing import Iterable, List, Union


def _convert(text: str) -> Union[int, str]:
    return int(text) if text.isdigit() else text


def _alphanumeric_key(key: str) -> List[Union[int, str]]:
    return [_convert(c) for c in re.split("([0-9]+)", key)]


def sorted_alphanumeric(iterable: Iterable[str]) -> Iterable[str]:
    """Sort the given iterable in the way that humans expect. For example, given ``{"class_1", "class_11",
    "class_2"}`` this returns ``["class_1", "class_2", "class_11"]``.

    Copied from:
    https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/
    """
    return sorted(iterable, key=_alphanumeric_key)
