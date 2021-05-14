from importlib.util import find_spec


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

_TEXT_AVAILABLE = _TRANSFORMERS_AVAILABLE
_TABULAR_AVAILABLE = _TABNET_AVAILABLE and _PANDAS_AVAILABLE
_VIDEO_AVAILABLE = _PYTORCHVIDEO_AVAILABLE
_IMAGE_AVAILABLE = _TORCHVISION_AVAILABLE and _TIMM_AVAILABLE and _KORNIA_AVAILABLE
