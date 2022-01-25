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
import functools
import importlib
import operator
import os
import types
from importlib.util import find_spec
from typing import List, Tuple, Union

from pkg_resources import DistributionNotFound

try:
    from packaging.version import Version
except (ModuleNotFoundError, DistributionNotFound):
    Version = None


def _module_available(module_path: str) -> bool:
    """Check if a path is available in your environment.

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
    except ValueError:
        # Sometimes __spec__ can be None and gives a ValueError
        return True


def _compare_version(package: str, op, version) -> bool:
    """Compare package version with some requirements.

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
        # this is mock by sphinx, so it shall return True to generate all summaries
        return True
    return op(pkg_version, Version(version))


_TORCH_AVAILABLE = _module_available("torch")
_PL_AVAILABLE = _module_available("pytorch_lightning")
_BOLTS_AVAILABLE = _module_available("pl_bolts") and _compare_version("torch", operator.lt, "1.9.0")
_PANDAS_AVAILABLE = _module_available("pandas")
_SKLEARN_AVAILABLE = _module_available("sklearn")
_PYTORCHTABULAR_AVAILABLE = _module_available("pytorch_tabular")
_FORECASTING_AVAILABLE = _module_available("pytorch_forecasting")
_KORNIA_AVAILABLE = _module_available("kornia")
_COCO_AVAILABLE = _module_available("pycocotools")
_TIMM_AVAILABLE = _module_available("timm")
_TORCHVISION_AVAILABLE = _module_available("torchvision")
_PYTORCHVIDEO_AVAILABLE = _module_available("pytorchvideo")
_MATPLOTLIB_AVAILABLE = _module_available("matplotlib")
_TRANSFORMERS_AVAILABLE = _module_available("transformers")
_PYSTICHE_AVAILABLE = _module_available("pystiche")
_FIFTYONE_AVAILABLE = _module_available("fiftyone")
_FASTAPI_AVAILABLE = _module_available("fastapi")
_PYDANTIC_AVAILABLE = _module_available("pydantic")
_GRAPHVIZ_AVAILABLE = _module_available("graphviz")
_CYTOOLZ_AVAILABLE = _module_available("cytoolz")
_UVICORN_AVAILABLE = _module_available("uvicorn")
_PIL_AVAILABLE = _module_available("PIL")
_OPEN3D_AVAILABLE = _module_available("open3d")
_SEGMENTATION_MODELS_AVAILABLE = _module_available("segmentation_models_pytorch")
_FASTFACE_AVAILABLE = _module_available("fastface") and _compare_version("pytorch_lightning", operator.lt, "1.5.0")
_LIBROSA_AVAILABLE = _module_available("librosa")
_TORCH_SCATTER_AVAILABLE = _module_available("torch_scatter")
_TORCH_SPARSE_AVAILABLE = _module_available("torch_sparse")
_TORCH_GEOMETRIC_AVAILABLE = _module_available("torch_geometric")
_NETWORKX_AVAILABLE = _module_available("networkx")
_TORCHAUDIO_AVAILABLE = _module_available("torchaudio")
_SENTENCEPIECE_AVAILABLE = _module_available("sentencepiece")
_DATASETS_AVAILABLE = _module_available("datasets")
_TM_TEXT_AVAILABLE: bool = _module_available("torchmetrics.text")
_ICEVISION_AVAILABLE = _module_available("icevision")
_ICEDATA_AVAILABLE = _module_available("icedata")
_LEARN2LEARN_AVAILABLE = _module_available("learn2learn") and _compare_version("learn2learn", operator.ge, "0.1.6")
_TORCH_ORT_AVAILABLE = _module_available("torch_ort")
_VISSL_AVAILABLE = _module_available("vissl") and _module_available("classy_vision")
_ALBUMENTATIONS_AVAILABLE = _module_available("albumentations")
_BAAL_AVAILABLE = _module_available("baal")
_TORCH_OPTIMIZER_AVAILABLE = _module_available("torch_optimizer")
_SENTENCE_TRANSFORMERS_AVAILABLE = _module_available("sentence_transformers")


if _PIL_AVAILABLE:
    from PIL import Image  # noqa: F401
else:

    class Image:
        Image = object


if Version:
    _TORCHVISION_GREATER_EQUAL_0_9 = _compare_version("torchvision", operator.ge, "0.9.0")
    _PL_GREATER_EQUAL_1_4_3 = _compare_version("pytorch_lightning", operator.ge, "1.4.3")
    _PL_GREATER_EQUAL_1_4_0 = _compare_version("pytorch_lightning", operator.ge, "1.4.0")
    _PL_GREATER_EQUAL_1_5_0 = _compare_version("pytorch_lightning", operator.ge, "1.5.0")
    _PANDAS_GREATER_EQUAL_1_3_0 = _compare_version("pandas", operator.ge, "1.3.0")
    _ICEVISION_GREATER_EQUAL_0_11_0 = _compare_version("icevision", operator.ge, "0.11.0")
    _TM_GREATER_EQUAL_0_7_0 = _compare_version("torchmetrics", operator.ge, "0.7.0")

_TEXT_AVAILABLE = all(
    [
        _TRANSFORMERS_AVAILABLE,
        _SENTENCEPIECE_AVAILABLE,
        _DATASETS_AVAILABLE,
        _TM_TEXT_AVAILABLE,
        _SENTENCE_TRANSFORMERS_AVAILABLE,
    ]
)
_TABULAR_AVAILABLE = _PANDAS_AVAILABLE and _FORECASTING_AVAILABLE and _PYTORCHTABULAR_AVAILABLE
_VIDEO_AVAILABLE = _TORCHVISION_AVAILABLE and _PIL_AVAILABLE and _PYTORCHVIDEO_AVAILABLE and _KORNIA_AVAILABLE
_IMAGE_AVAILABLE = all(
    [
        _TORCHVISION_AVAILABLE,
        _TIMM_AVAILABLE,
        _PIL_AVAILABLE,
        _KORNIA_AVAILABLE,
        _PYSTICHE_AVAILABLE,
        _SEGMENTATION_MODELS_AVAILABLE,
    ]
)
_SERVE_AVAILABLE = _FASTAPI_AVAILABLE and _PYDANTIC_AVAILABLE and _CYTOOLZ_AVAILABLE and _UVICORN_AVAILABLE
_POINTCLOUD_AVAILABLE = _OPEN3D_AVAILABLE and _TORCHVISION_AVAILABLE
_AUDIO_AVAILABLE = all([_TORCHAUDIO_AVAILABLE, _LIBROSA_AVAILABLE, _TRANSFORMERS_AVAILABLE])
_GRAPH_AVAILABLE = (
    _TORCH_SCATTER_AVAILABLE and _TORCH_SPARSE_AVAILABLE and _TORCH_GEOMETRIC_AVAILABLE and _NETWORKX_AVAILABLE
)

_EXTRAS_AVAILABLE = {
    "image": _IMAGE_AVAILABLE,
    "tabular": _TABULAR_AVAILABLE,
    "text": _TEXT_AVAILABLE,
    "video": _VIDEO_AVAILABLE,
    "pointcloud": _POINTCLOUD_AVAILABLE,
    "serve": _SERVE_AVAILABLE,
    "audio": _AUDIO_AVAILABLE,
    "graph": _GRAPH_AVAILABLE,
}


def requires(module_paths: Union[str, Tuple[bool, str], List[Union[str, Tuple[bool, str]]]]):

    if not isinstance(module_paths, list):
        module_paths = [module_paths]

    def decorator(func):
        available = True
        extras = []
        modules = []
        for module_path in module_paths:
            if isinstance(module_path, str):
                if module_path in _EXTRAS_AVAILABLE:
                    extras.append(module_path)
                    if not _EXTRAS_AVAILABLE[module_path]:
                        available = False
                else:
                    modules.append(module_path)
                    if not _module_available(module_path):
                        available = False
            else:
                available, module_path = module_path
                modules.append(module_path)

        if not available:
            modules = [f"'{module}'" for module in modules]
            modules.append(f"'lightning-flash[{','.join(extras)}]'")

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                raise ModuleNotFoundError(
                    f"Required dependencies not available. Please run: pip install {' '.join(modules)}"
                )

            return wrapper
        return func

    return decorator


def example_requires(module_paths: Union[str, List[str]]):
    return requires(module_paths)(lambda: None)()


def lazy_import(module_name, callback=None):
    """Returns a proxy module object that will lazily import the given module the first time it is used.

    Example usage::

        # Lazy version of `import tensorflow as tf`
        tf = lazy_import("tensorflow")

        # Other commands

        # Now the module is loaded
        tf.__version__

    Args:
        module_name: the fully-qualified module name to import
        callback (None): a callback function to call before importing the
            module

    Returns:
        a proxy module object that will be lazily imported when first used
    """
    return LazyModule(module_name, callback=callback)


class LazyModule(types.ModuleType):
    """Proxy module that lazily imports the underlying module the first time it is actually used.

    Args:
        module_name: the fully-qualified module name to import
        callback (None): a callback function to call before importing the
            module
    """

    def __init__(self, module_name, callback=None):
        super().__init__(module_name)
        self._module = None
        self._callback = callback

    def __getattr__(self, item):
        if self._module is None:
            self._import_module()

        return getattr(self._module, item)

    def __dir__(self):
        if self._module is None:
            self._import_module()

        return dir(self._module)

    def _import_module(self):
        # Execute callback, if any
        if self._callback is not None:
            self._callback()

        # Actually import the module
        module = importlib.import_module(self.__name__)
        self._module = module

        # Update this object's dict so that attribute references are efficient
        # (__getattr__ is only called on lookups that fail)
        self.__dict__.update(module.__dict__)


# Global variables used for testing purposes (e.g. to only run doctests in the correct CI job)
_IMAGE_TESTING = _IMAGE_AVAILABLE
_IMAGE_EXTRAS_TESTING = False  # Not for normal use
_VIDEO_TESTING = _VIDEO_AVAILABLE
_VIDEO_EXTRAS_TESTING = False  # Not for normal use
_TABULAR_TESTING = _TABULAR_AVAILABLE
_TEXT_TESTING = _TEXT_AVAILABLE
_SERVE_TESTING = _SERVE_AVAILABLE
_POINTCLOUD_TESTING = _POINTCLOUD_AVAILABLE
_GRAPH_TESTING = _GRAPH_AVAILABLE
_AUDIO_TESTING = _AUDIO_AVAILABLE

if "FLASH_TEST_TOPIC" in os.environ:
    topic = os.environ["FLASH_TEST_TOPIC"]
    _IMAGE_TESTING = topic == "image"
    _IMAGE_EXTRAS_TESTING = topic == "image,image_extras"
    _VIDEO_TESTING = topic == "video"
    _VIDEO_EXTRAS_TESTING = topic == "video,video_extras"
    _TABULAR_TESTING = topic == "tabular"
    _TEXT_TESTING = topic == "text"
    _SERVE_TESTING = topic == "serve"
    _POINTCLOUD_TESTING = topic == "pointcloud"
    _GRAPH_TESTING = topic == "graph"
    _AUDIO_TESTING = topic == "audio"
