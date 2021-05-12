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
"""Root package info."""
import os

from flash.__about__ import *  # noqa: F401 F403

_PACKAGE_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

from flash.core.model import Task  # noqa: E402
from flash.core.trainer import Trainer  # noqa: E402
from flash.data.data_module import DataModule  # noqa: E402

__all__ = [
    "Task",
    "Trainer",
    "DataModule",
]
