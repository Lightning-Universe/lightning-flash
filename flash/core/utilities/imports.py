from pytorch_lightning.utilities.imports import _module_available

_TORCH_AVAILABLE = _module_available("torch")
_PANDAS_AVAILABLE = _module_available("pandas")
_TABNET_AVAILABLE = _module_available("pytorch_tabnet")
_KORNIA_AVAILABLE = _module_available("kornia")
_COCO_AVAILABLE = _module_available("pycocotools")
_TIMM_AVAILABLE = _module_available("timm")
_TORCHVISION_AVAILABLE = _module_available("torchvision")
_PYTORCHVIDEO_AVAILABLE = _module_available("pytorchvideo")
_MATPLOTLIB_AVAILABLE = _module_available("matplotlib")
_TRANSFORMERS_AVAILABLE = _module_available("transformers")

_TEXT_AVAILABLE = _TRANSFORMERS_AVAILABLE
_TABULAR_AVAILABLE = _TABNET_AVAILABLE
_VIDEO_AVAILABLE = _PYTORCHVIDEO_AVAILABLE
_IMAGE_AVAILABLE = _TIMM_AVAILABLE
