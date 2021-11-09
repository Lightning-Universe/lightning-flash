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
from typing import Tuple
from unittest import mock

import numpy as np
import pytest
import torch

from flash import Trainer
from flash.__main__ import main
from flash.core.data.data_pipeline import DataPipeline
from flash.core.data.io.input import InputDataKeys
from flash.core.utilities.imports import _IMAGE_AVAILABLE
from flash.image import SemanticSegmentation
from flash.image.segmentation.data import SemanticSegmentationInputTransform
from tests.helpers.utils import _IMAGE_TESTING, _SERVE_TESTING

# ======== Mock functions ========


class DummyDataset(torch.utils.data.Dataset):
    size: Tuple[int, int] = (224, 224)
    num_classes: int = 8

    def __getitem__(self, index):
        return {
            InputDataKeys.INPUT: torch.rand(3, *self.size),
            InputDataKeys.TARGET: torch.randint(self.num_classes - 1, self.size),
        }

    def __len__(self) -> int:
        return 10


# ==============================


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_smoke():
    model = SemanticSegmentation(num_classes=1)
    assert model is not None


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
@pytest.mark.parametrize("num_classes", [8, 256])
@pytest.mark.parametrize("img_shape", [(1, 3, 224, 192), (2, 3, 128, 256)])
def test_forward(num_classes, img_shape):
    model = SemanticSegmentation(
        num_classes=num_classes,
        backbone="resnet50",
        head="fpn",
    )

    B, C, H, W = img_shape
    img = torch.rand(B, C, H, W)

    out = model(img)
    assert out.shape == (B, num_classes, H, W)


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_init_train(tmpdir):
    model = SemanticSegmentation(num_classes=10)
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.finetune(model, train_dl, strategy="freeze_unfreeze")


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_non_existent_backbone():
    with pytest.raises(KeyError):
        SemanticSegmentation(2, "i am never going to implement this lol")


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_freeze():
    model = SemanticSegmentation(2)
    model.freeze()
    for p in model.backbone.parameters():
        assert p.requires_grad is False


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_unfreeze():
    model = SemanticSegmentation(2)
    model.unfreeze()
    for p in model.backbone.parameters():
        assert p.requires_grad is True


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_predict_tensor():
    img = torch.rand(1, 3, 64, 64)
    model = SemanticSegmentation(2, backbone="mobilenetv3_large_100")
    data_pipe = DataPipeline(input_transform=SemanticSegmentationInputTransform(num_classes=1))
    out = model.predict(img, ="tensors", data_pipeline=data_pipe)
    assert isinstance(out[0], list)
    assert len(out[0]) == 64
    assert len(out[0][0]) == 64


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_predict_numpy():
    img = np.ones((1, 3, 64, 64))
    model = SemanticSegmentation(2, backbone="mobilenetv3_large_100")
    data_pipe = DataPipeline(input_transform=SemanticSegmentationInputTransform(num_classes=1))
    out = model.predict(img, ="numpy", data_pipeline=data_pipe)
    assert isinstance(out[0], list)
    assert len(out[0]) == 64
    assert len(out[0][0]) == 64


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
@pytest.mark.parametrize("jitter, args", [(torch.jit.trace, (torch.rand(1, 3, 32, 32),))])
def test_jit(tmpdir, jitter, args):
    path = os.path.join(tmpdir, "test.pt")

    model = SemanticSegmentation(2)
    model.eval()

    model = jitter(model, *args)

    torch.jit.save(model, path)
    model = torch.jit.load(path)

    out = model(torch.rand(1, 3, 32, 32))
    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([1, 2, 32, 32])


@pytest.mark.skipif(not _SERVE_TESTING, reason="serve libraries aren't installed.")
@mock.patch("flash._IS_TESTING", True)
def test_serve():
    model = SemanticSegmentation(2)
    # TODO: Currently only servable once a input_transform has been attached
    model._input_transform = SemanticSegmentationInputTransform()
    model.eval()
    model.serve()


@pytest.mark.skipif(_IMAGE_AVAILABLE, reason="image libraries are installed.")
def test_load_from_checkpoint_dependency_error():
    with pytest.raises(ModuleNotFoundError, match=re.escape("'lightning-flash[image]'")):
        SemanticSegmentation.load_from_checkpoint("not_a_real_checkpoint.pt")


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_available_pretrained_weights():
    assert SemanticSegmentation.available_pretrained_weights("resnet18") == ["imagenet", "ssl", "swsl"]


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_cli():
    cli_args = ["flash", "semantic_segmentation", "--trainer.fast_dev_run", "True"]
    with mock.patch("sys.argv", cli_args):
        try:
            main()
        except SystemExit:
            pass
