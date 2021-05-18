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
"""General utilities"""

import importlib
import operator
from importlib.util import find_spec

from pkg_resources import DistributionNotFound

try:
    from packaging.version import Version
except (ModuleNotFoundError, DistributionNotFound):
    Version = None


def _module_available(module_path: str) -> bool:
    """
    Check if a path is available in your environment

    >>> _module_available('os')
    True
    >>> _module_available('bla.bla')
    False
    """
    try:
        return find_spec(module_path) is not None
    except AttributeError:
        # Python 3.6
        return False
    except ModuleNotFoundError:
        # Python 3.7+
        return False


def _compare_version(package: str, op, version) -> bool:
    """
    Compare package version with some requirements

    >>> _compare_version("torch", operator.ge, "0.1")
    True
    """
    try:
        pkg = importlib.import_module(package)
    except (ModuleNotFoundError, DistributionNotFound, ValueError):
        return False
    try:
        pkg_version = Version(pkg.__version__)
    except TypeError:
        # this is mock by sphinx, so it shall return True ro generate all summaries
        return True
    return op(pkg_version, Version(version))


_TORCH_AVAILABLE = _module_available("torch")
_PANDAS_AVAILABLE = _module_available("pandas")
_SKLEARN_AVAILABLE = _module_available("sklearn")
_TABNET_AVAILABLE = _module_available("pytorch_tabnet")
_KORNIA_AVAILABLE = _module_available("kornia")
_COCO_AVAILABLE = _module_available("pycocotools")
_TIMM_AVAILABLE = _module_available("timm")
_TORCHVISION_AVAILABLE = _module_available("torchvision")
_PYTORCHVIDEO_AVAILABLE = _module_available("pytorchvideo")
_MATPLOTLIB_AVAILABLE = _module_available("matplotlib")
_TRANSFORMERS_AVAILABLE = _module_available("transformers")
_PYSTICHE_AVAILABLE = _module_available("pystiche")

if Version:
    _TORCHVISION_GREATER_EQUAL_0_9 = _compare_version("torchvision", operator.ge, "0.9.0")
    _PYSTICHE_GREATER_EQUAL_0_7_2 = _compare_version("pystiche", operator.ge, "0.7.2")

_IMAGE_STLYE_TRANSFER = _PYSTICHE_AVAILABLE
_TEXT_AVAILABLE = _TRANSFORMERS_AVAILABLE
_TABULAR_AVAILABLE = _TABNET_AVAILABLE and _PANDAS_AVAILABLE
_VIDEO_AVAILABLE = _PYTORCHVIDEO_AVAILABLE
_IMAGE_AVAILABLE = _TORCHVISION_AVAILABLE and _TIMM_AVAILABLE and _KORNIA_AVAILABLE
