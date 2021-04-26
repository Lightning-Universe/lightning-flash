import pytest
import torch

from flash.vision import SemanticSegmentation


def test_smoke():
    model = SemanticSegmentation(num_classes=1)
    assert model is not None


@pytest.mark.parametrize("num_classes", [8, 256])
@pytest.mark.parametrize("img_shape", [(1, 3, 224, 192), (2, 3, 127, 212)])
def test_forward(num_classes, img_shape):
    model = SemanticSegmentation(
        num_classes=num_classes,
        backbone='torchvision/fcn_resnet50',
    )

    B, C, H, W = img_shape
    img = torch.rand(B, C, H, W)

    out = model(img)
    assert out.shape == (B, num_classes, H, W)
