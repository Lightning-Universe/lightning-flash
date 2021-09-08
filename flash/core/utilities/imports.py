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
import types
from importlib.util import find_spec
from typing import List, Union
from warnings import warn

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
_BOLTS_AVAILABLE = _module_available("pl_bolts") and _compare_version("torch", operator.lt, "1.9.0")
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
_FIFTYONE_AVAILABLE = _module_available("fiftyone")
_FASTAPI_AVAILABLE = _module_available("fastapi")
_PYDANTIC_AVAILABLE = _module_available("pydantic")
_GRAPHVIZ_AVAILABLE = _module_available("graphviz")
_CYTOOLZ_AVAILABLE = _module_available("cytoolz")
_UVICORN_AVAILABLE = _module_available("uvicorn")
_PIL_AVAILABLE = _module_available("PIL")
_OPEN3D_AVAILABLE = _module_available("open3d")
_SEGMENTATION_MODELS_AVAILABLE = _module_available("segmentation_models_pytorch")
_LIBROSA_AVAILABLE = _module_available("librosa")
_TORCH_SCATTER_AVAILABLE = _module_available("torch_scatter")
_TORCH_SPARSE_AVAILABLE = _module_available("torch_sparse")
_TORCH_GEOMETRIC_AVAILABLE = _module_available("torch_geometric")
_TORCHAUDIO_AVAILABLE = _module_available("torchaudio")
_ROUGE_SCORE_AVAILABLE = _module_available("rouge_score")
_SENTENCEPIECE_AVAILABLE = _module_available("sentencepiece")
_DATASETS_AVAILABLE = _module_available("datasets")
_ICEVISION_AVAILABLE = _module_available("icevision")
_ICEDATA_AVAILABLE = _module_available("icedata")
_TORCH_ORT_AVAILABLE = _module_available("torch_ort")
_VISSL_AVAILABLE = _module_available("vissl") and _module_available("classy_vision")

if _PIL_AVAILABLE:
    from PIL import Image
else:

    class MetaImage(type):
        def __init__(cls, name, bases, dct):
            super().__init__(name, bases, dct)

            cls._Image = None

        @property
        def Image(cls):
            warn("Mock object called due to missing PIL library. Please use \"pip install 'lightning-flash[image]'\".")
            return cls._Image

    class Image(metaclass=MetaImage):
        pass


if Version:
    _TORCHVISION_GREATER_EQUAL_0_9 = _compare_version("torchvision", operator.ge, "0.9.0")
    _PL_GREATER_EQUAL_1_4_3 = _compare_version("pytorch_lightning", operator.ge, "1.4.3")

_TEXT_AVAILABLE = all(
    [
        _TRANSFORMERS_AVAILABLE,
        _ROUGE_SCORE_AVAILABLE,
        _SENTENCEPIECE_AVAILABLE,
        _DATASETS_AVAILABLE,
    ]
)
_TABULAR_AVAILABLE = _TABNET_AVAILABLE and _PANDAS_AVAILABLE
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
_GRAPH_AVAILABLE = _TORCH_SCATTER_AVAILABLE and _TORCH_SPARSE_AVAILABLE and _TORCH_GEOMETRIC_AVAILABLE

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


def requires(module_paths: Union[str, List]):

    if not isinstance(module_paths, list):
        module_paths = [module_paths]

    def decorator(func):
        available = True
        extras = []
        modules = []
        for module_path in module_paths:
            if module_path in _EXTRAS_AVAILABLE:
                extras.append(module_path)
                if not _EXTRAS_AVAILABLE[module_path]:
                    available = False
            else:
                modules.append(module_path)
                if not _module_available(module_path):
                    available = False

        if not available:
            modules = [f"'{module}'" for module in modules]
            modules.append(f"'lightning-flash[{','.join(extras)}]'")

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                raise ModuleNotFoundError(
                    f"Required dependencies not available. Please run: pip install {' '.join(modules)}"
                )

            return wrapper
        else:
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
