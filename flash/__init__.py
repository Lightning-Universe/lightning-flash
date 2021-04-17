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

__version__ = "0.2.3"
__author__ = "PyTorchLightning et al."
__author_email__ = "name@pytorchlightning.ai"
__license__ = 'Apache-2.0'
__copyright__ = f"Copyright (c) 2020-2021, f{__author__}."
__homepage__ = "https://github.com/PyTorchLightning/lightning-flash"
__docs_url__ = "https://lightning-flash.readthedocs.io/en/stable/"
__docs__ = "Flash is a framework for fast prototyping, finetuning, and solving most standard deep learning challenges"
__long_doc__ = """
Flash is a task-based deep learning framework for flexible deep learning built on PyTorch Lightning.
Tasks can be anything from text classification to object segmentation.
Although PyTorch Lightning provides ultimate flexibility, for common tasks it does not remove 100% of the boilerplate.
Flash is built for applied researchers, beginners, data scientists, Kagglers or anyone starting out with Deep Learning.
But unlike other entry-level frameworks (keras, etc...), Flash users can switch to Lightning trivially when they need
the added flexibility.
"""

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

try:
    # This variable is injected in the __builtins__ by the build process.
    # It used to enable importing subpackages when the binaries are not built.
    __LIGHTNING_FLASH_SETUP__
except NameError:
    __LIGHTNING_FLASH_SETUP__: bool = False

if __LIGHTNING_FLASH_SETUP__:
    import sys  # pragma: no-cover

    sys.stdout.write(f"Partial import of `{__name__}` during the build process.\n")  # pragma: no-cover
    # We are not importing the rest of the lightning during the build process, as it may not be compiled yet
else:

    from flash import tabular, text, vision
    from flash.core import data, utils
    from flash.core.classification import ClassificationTask
    from flash.core.data import DataModule
    from flash.core.data.utils import download_data
    from flash.core.model import Task
    from flash.core.trainer import Trainer

    __all__ = [
        "Task",
        "ClassificationTask",
        "DataModule",
        "vision",
        "text",
        "tabular",
        "data",
        "utils",
        "download_data",
    ]
