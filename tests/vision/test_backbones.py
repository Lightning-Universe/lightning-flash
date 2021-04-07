import pytest
from pytorch_lightning.utilities import _BOLTS_AVAILABLE, _TORCHVISION_AVAILABLE

from flash.utils.imports import _TIMM_AVAILABLE
from flash.vision.backbones import IMAGE_CLASSIFIER_BACKBONES


@pytest.mark.parametrize(["backbone", "expected_num_features"], [
    pytest.param("resnet34", 512, marks=pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="No torchvision")),
    pytest.param("mobilenetv2_100", 1280, marks=pytest.mark.skipif(not _TIMM_AVAILABLE, reason="No timm")),
    pytest.param("simclr-imagenet", 2048, marks=pytest.mark.skipif(not _BOLTS_AVAILABLE, reason="No bolts")),
    pytest.param("swav-imagenet", 2048, marks=pytest.mark.skipif(not _BOLTS_AVAILABLE, reason="No bolts")),
    pytest.param("mobilenet_v2", 1280, marks=pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="No torchvision")),
])
def test_image_classifier_backbones_registry(backbone, expected_num_features):
    backbone_fn = IMAGE_CLASSIFIER_BACKBONES.get(backbone)
    backbone_model, num_features = backbone_fn(pretrained=False)
    assert backbone_model
    assert num_features == expected_num_features
