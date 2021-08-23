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

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _PYSTICHE_AVAILABLE
from flash.core.utilities.providers import _PYSTICHE

STYLE_TRANSFER_BACKBONES = FlashRegistry("backbones")

__all__ = ["STYLE_TRANSFER_BACKBONES"]

if _PYSTICHE_AVAILABLE:

    from pystiche import enc

    MLE_FN_PATTERN = re.compile(r"^(?P<name>\w+?)_multi_layer_encoder$")

    for mle_fn in dir(enc):
        match = MLE_FN_PATTERN.match(mle_fn)
        if not match:
            continue

        STYLE_TRANSFER_BACKBONES(
            fn=lambda: (getattr(enc, mle_fn)(), None),
            name=match.group("name"),
            namespace="image/style_transfer",
            providers=_PYSTICHE,
        )
