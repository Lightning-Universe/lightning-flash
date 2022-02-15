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
from flash.core.utilities.imports import _TORCH_AVAILABLE

if _TORCH_AVAILABLE:

    from flash.core.data.callback import FlashCallback
    from flash.core.data.data_module import DataModule
    from flash.core.data.io.input import DataKeys, Input
    from flash.core.data.io.input_transform import InputTransform
    from flash.core.data.io.output import Output
    from flash.core.data.io.output_transform import OutputTransform
    from flash.core.model import Task
    from flash.core.trainer import Trainer
    from flash.core.utilities.stages import RunningStage

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
