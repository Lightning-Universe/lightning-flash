import pytest

from flash.vision.backbones import backbone_and_num_features


@pytest.mark.parametrize(["backbone", "expected_num_features"], [("resnet34", 512), ("mobilenet_v2", 1280),
                                                                 ("simclr-imagenet", 2048), ("vgg16", 512)])
def test_backbone_and_num_features(backbone, expected_num_features):

    backbone_model, num_features = backbone_and_num_features(model_name=backbone, pretrained=False, fpn=False)

    assert backbone_model
    assert num_features == expected_num_features
