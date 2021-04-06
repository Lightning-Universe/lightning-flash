from pytorch_lightning.utilities.imports import _module_available

_TABNET_AVAILABLE = _module_available("pytorch_tabnet")
_KORNIA_AVAILABLE = _module_available("kornia")
_COCO_AVAILABLE = _module_available("pycocotools")
_TIMM_AVAILABLE = _module_available("timm")
