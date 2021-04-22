import pytest
import torch

from flash.vision import SemanticSegmentation


def test_smoke():
    model = SemanticSegmentation(num_classes=1)
    assert model is not None


@pytest.mark.skip(reason="todo")
def test_forward():
    num_classes = 5
    model = SemanticSegmentation(num_classes)

    img = torch.rand(1, 3, 224, 224)
    out = model(img)
    assert out.shape == (1, num_classes, 224, 224)
