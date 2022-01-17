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
from typing import Any, Dict, List, Optional, TypeVar

from flash.core.data.io.input import DataKeys
from flash.core.data.utilities.classification import _is_list_like

T = TypeVar("T")


def to_sample(input: Any) -> Dict[str, Any]:
    """Cast a single input to a sample dictionary. Uses the following rules:

    * If the input is a dictionary with an "input" key, it will be returned
    * If the input is list-like and of length 2 then the first element will be treated as the input and the second
        element will be treated as the target
    * Else the whole input will be mapped by the input key in the returned sample

    Args:
        input: The input to cast to a sample.

    Returns:
        A sample dictionary.
    """
    if isinstance(input, dict) and DataKeys.INPUT in input:
        return input
    if _is_list_like(input) and len(input) == 2:
        if input[1] is not None:
            return {DataKeys.INPUT: input[0], DataKeys.TARGET: input[1]}
        input = input[0]
    return {DataKeys.INPUT: input}


def to_samples(inputs: List[Any], targets: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    """Package a list of inputs and, optionally, a list of targets in a list of dictionaries (samples).

    Args:
        inputs: The list of inputs to package as dictionaries.
        targets: Optionally provide a list of targets to also be included in the samples.

    Returns:
        A list of sample dictionaries.
    """
    if targets is None:
        return [to_sample(input) for input in inputs]
    return [to_sample(input) for input in zip(inputs, targets)]
