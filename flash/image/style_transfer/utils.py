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
from typing import NoReturn

__all__ = ["raise_not_supported"]


def raise_not_supported(phase: str) -> NoReturn:
    raise RuntimeError(
        f"Style transfer does not support a {phase} phase, "
        f"since there is no metric to objectively determine the quality of a stylization."
    )
