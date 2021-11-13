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
import warnings

from pytorch_lightning.utilities import LightningEnum

from flash.core.utilities.on_access_enum_meta import OnAccessEnumMeta


class DefaultDataKeys(LightningEnum, metaclass=OnAccessEnumMeta):
    """Deprecated since 0.6.0 and will be removed in 0.7.0.

    Use `flash.DataKeys` instead.
    """

    INPUT = "input"
    PREDS = "preds"
    TARGET = "target"
    METADATA = "metadata"

    def __new__(cls, value):
        member = str.__new__(cls, value)
        member._on_access = member.deprecate
        return member

    def deprecate(self):
        warnings.warn(
            "`DefaultDataKeys` was deprecated in 0.6.0 and will be removed in 0.7.0. Use `flash.DataKeys` instead.",
            FutureWarning,
        )

    # TODO: Create a FlashEnum class???
    def __hash__(self) -> int:
        return hash(self.value)
