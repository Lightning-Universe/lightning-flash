from pytorch_lightning.utilities.imports import _module_available

_TABNET_AVAILABLE = _module_available("pytorch_tabnet")
_KORNIA_AVAILABLE = _module_available("kornia")
_COCO_AVAILABLE = _module_available("pycocotools")
_TIMM_AVAILABLE = _module_available("timm")
_TORCHVISION_AVAILABLE = _module_available("torchvision")
_PYTORCHVIDEO_AVAILABLE = _module_available("pytorchvideo")
_MATPLOTLIB_AVAILABLE = _module_available("matplotlib")
_TRANSFORMERS_AVAILABLE = _module_available("transformers")
_PYSTICHE_AVAILABLE = _module_available("pystiche")
