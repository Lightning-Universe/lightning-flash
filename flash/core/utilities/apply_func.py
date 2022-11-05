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
from typing import Callable, Dict, Mapping, Sequence, Union

from torch import nn


def get_callable_name(fn_or_class: Union[Callable, object]) -> str:
    return getattr(fn_or_class, "__name__", fn_or_class.__class__.__name__).lower()


def get_callable_dict(fn: Union[nn.Module, Callable, Mapping, Sequence]) -> Union[Dict, Mapping]:
    if isinstance(fn, nn.Module):
        return nn.ModuleDict({get_callable_name(fn): fn})
    if isinstance(fn, Mapping):
        return fn
    if isinstance(fn, Sequence):
        return {get_callable_name(f): f for f in fn}
    if callable(fn):
        return {get_callable_name(fn): fn}
