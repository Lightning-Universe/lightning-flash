from pytorch_lightning.utilities.imports import _module_available

_BOLTS_AVAILABLE = _module_available("pl_bolts")
_TABNET_AVAILABLE = _module_available("pytorch_tabnet")
_TORCHVISION_AVAILABLE = _module_available('torchvision')
_KORNIA_AVAILABLE = _module_available("kornia")
_COCO_AVAILABLE = _module_available("pycocotools")
_TIMM_AVAILABLE = _module_available("timm")
