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

# import numpy
#
# # adding compatibility for numpy >= 1.24
# for tp_name, tp_ins in [("object", object), ("bool", float), ("int", int), ("float", float)]:
#     if not hasattr(numpy, tp_name):
#         setattr(numpy, tp_name, tp_ins)

from flash.__about__ import *  # noqa: F401 E402 F403
from flash.core.data.callback import FlashCallback  # noqa: E402
from flash.core.data.data_module import DataModule  # noqa: E402
from flash.core.data.io.input import DataKeys, Input  # noqa: E402
from flash.core.data.io.input_transform import InputTransform  # noqa: E402
from flash.core.data.io.output import Output  # noqa: E402
from flash.core.data.io.output_transform import OutputTransform  # noqa: E402
from flash.core.model import Task  # noqa: E402
from flash.core.trainer import Trainer  # noqa: E402
from flash.core.utilities.stages import RunningStage  # noqa: E402

_PACKAGE_ROOT = os.path.dirname(__file__)
ASSETS_ROOT = os.path.join(_PACKAGE_ROOT, "assets")
PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)
_IS_TESTING = os.getenv("FLASH_TESTING", "0") == "1"

if _IS_TESTING:
    from pytorch_lightning import seed_everything

    seed_everything(42)

__all__ = [
    "DataKeys",
    "DataModule",
    "FlashCallback",
    "Input",
    "InputTransform",
    "Output",
    "OutputTransform",
    "RunningStage",
    "Task",
    "Trainer",
]
