import os
import re

import pytest
import torch

from flash.core.utilities.imports import _IMAGE_AVAILABLE
from flash.image.style_transfer import StyleTransfer
from tests.helpers.utils import _IMAGE_TESTING


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_style_transfer_task():

    model = StyleTransfer(
        backbone="vgg11", content_layer="relu1_2", content_weight=10, style_layers="relu1_2", style_weight=11
    )
    assert model.perceptual_loss.content_loss.encoder.layer == "relu1_2"
    assert model.perceptual_loss.content_loss.score_weight == 10
    assert "relu1_2" in [n for n, m in model.perceptual_loss.style_loss.named_modules()]
    assert model.perceptual_loss.style_loss.score_weight == 11


@pytest.mark.skipif(_IMAGE_AVAILABLE, reason="image libraries are installed.")
def test_style_transfer_task_import():
    with pytest.raises(ModuleNotFoundError, match="[image]"):
        StyleTransfer()


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_jit(tmpdir):
    path = os.path.join(tmpdir, "test.pt")

    model = StyleTransfer()
    model.eval()

    model = torch.jit.trace(model, torch.rand(1, 3, 32, 32))  # torch.jit.script doesn't work with pystiche

    torch.jit.save(model, path)
    model = torch.jit.load(path)

    out = model(torch.rand(1, 3, 32, 32))
    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([1, 3, 32, 32])


@pytest.mark.skipif(_IMAGE_AVAILABLE, reason="image libraries are installed.")
def test_load_from_checkpoint_dependency_error():
    with pytest.raises(ModuleNotFoundError, match=re.escape("'lightning-flash[image]'")):
        StyleTransfer.load_from_checkpoint("not_a_real_checkpoint.pt")
