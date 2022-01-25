# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
from unittest import mock

import pytest
import torch

from flash.__main__ import main
from flash.core.utilities.imports import _IMAGE_AVAILABLE, _IMAGE_TESTING
from flash.image.style_transfer import StyleTransfer


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

    model.loss_fn = None
    model.perceptual_loss = None  # TODO: Document this

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


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_cli():
    cli_args = ["flash", "style_transfer", "--trainer.fast_dev_run", "True"]
    with mock.patch("sys.argv", cli_args):
        try:
            main()
        except SystemExit:
            pass
