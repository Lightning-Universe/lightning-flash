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


def _typed_isinstance(__object, __class_or_tuple):
    return isinstance(__object, getattr(__class_or_tuple, "__origin__", __class_or_tuple))


try:
    from torch.jit import isinstance as _isinstance
except ImportError:
    _isinstance = _typed_isinstance
