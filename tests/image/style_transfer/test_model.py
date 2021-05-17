import pytest

from flash.core.utilities.imports import _IMAGE_STLYE_TRANSFER
from flash.image.style_transfer import StyleTransfer


@pytest.mark.skipif(not _IMAGE_STLYE_TRANSFER, reason="image style transfer libraries aren't installed.")
def test_style_transfer_task():

    model = StyleTransfer(
        backbone="vgg11", content_layer="relu1_2", content_weight=10, style_layers="relu1_2", style_weight=11
    )
    assert model.perceptual_loss.content_loss.encoder.layer == "relu1_2"
    assert model.perceptual_loss.content_loss.score_weight == 10
    assert "relu1_2" in [n for n, m in model.perceptual_loss.style_loss.named_modules()]
    assert model.perceptual_loss.style_loss.score_weight == 11


@pytest.mark.skipif(_IMAGE_STLYE_TRANSFER, reason="image style transfer libraries are installed.")
def test_style_transfer_task_import():
    with pytest.raises(ModuleNotFoundError, match="[image_style_transfer]"):
        StyleTransfer()
