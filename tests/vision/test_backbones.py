import pytest
from pytorch_lightning.utilities import _BOLTS_AVAILABLE, _TORCHVISION_AVAILABLE

from flash.utils.imports import _TIMM_AVAILABLE
from flash.vision.backbones import IMAGE_CLASSIFIER_BACKBONES


@pytest.mark.parametrize(["backbone", "expected_num_features", "should_run"],
                         [("resnet34", 512, _TORCHVISION_AVAILABLE), ("mobilenetv2_100", 1280, _TIMM_AVAILABLE),
                          ("simclr-imagenet", 2048, _BOLTS_AVAILABLE), ("swav-imagenet", 2048, _BOLTS_AVAILABLE),
                          ("mobilenet_v2", 1280, _TORCHVISION_AVAILABLE)])
def test_image_classifier_backbones_registry(backbone, expected_num_features, should_run):

    if should_run:
        backbone_model, num_features = IMAGE_CLASSIFIER_BACKBONES.get(backbone)(pretrained=False)
        assert backbone_model
        assert num_features == expected_num_features
