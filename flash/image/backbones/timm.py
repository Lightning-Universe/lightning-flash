import torch.nn as nn

from functools import partial
from typing import Tuple

from flash.core.utilities.imports import _TIMM_AVAILABLE
from flash.image.backbones import IMAGE_CLASSIFIER_BACKBONES, TORCHVISION_MODELS
from flash.image.backbones.utilities import catch_url_error


if _TIMM_AVAILABLE:
    import timm

    def _fn_timm(
        model_name: str,
        pretrained: bool = True,
        num_classes: int = 0,
        **kwargs,
    ) -> Tuple[nn.Module, int]:
        backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, **kwargs)
        num_features = backbone.num_features
        return backbone, num_features

    for model_name in timm.list_models():

        if model_name in TORCHVISION_MODELS:
            continue

        IMAGE_CLASSIFIER_BACKBONES(
            fn=catch_url_error(partial(_fn_timm, model_name)), name=model_name, namespace="vision", package="timm"
        )
