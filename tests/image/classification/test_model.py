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

from flash import Trainer
from flash.__main__ import main
from flash.core.classification import Probabilities
from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _IMAGE_AVAILABLE
from flash.image import ImageClassifier
from flash.image.classification.data import ImageClassificationInputTransform
from tests.helpers.utils import _IMAGE_TESTING, _SERVE_TESTING

# ======== Mock functions ========


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return {
            DataKeys.INPUT: torch.rand(3, 224, 224),
            DataKeys.TARGET: torch.randint(10, size=(1,)).item(),
        }

    def __len__(self) -> int:
        return 100


class DummyMultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __getitem__(self, index):
        return {
            DataKeys.INPUT: torch.rand(3, 224, 224),
            DataKeys.TARGET: torch.randint(0, 2, (self.num_classes,)),
        }

    def __len__(self) -> int:
        return 100


# ==============================


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
@pytest.mark.parametrize(
    "backbone,metrics",
    [
        ("resnet18", None),
        ("resnet18", []),
        # "resnet34",
        # "resnet50",
        # "resnet101",
        # "resnet152",
    ],
)
def test_init_train(tmpdir, backbone, metrics):
    model = ImageClassifier(10, backbone=backbone, metrics=metrics)
    train_dl = torch.utils.data.DataLoader(DummyDataset())
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.finetune(model, train_dl, strategy="freeze")


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_non_existent_backbone():
    with pytest.raises(KeyError):
        ImageClassifier(2, "i am never going to implement this lol")


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_freeze():
    model = ImageClassifier(2)
    model.freeze()
    for p in model.backbone.parameters():
        assert p.requires_grad is False


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_unfreeze():
    model = ImageClassifier(2)
    model.unfreeze()
    for p in model.backbone.parameters():
        assert p.requires_grad is True


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_multilabel(tmpdir):

    num_classes = 4
    ds = DummyMultiLabelDataset(num_classes)
    model = ImageClassifier(num_classes, multi_label=True, output=Probabilities(multi_label=True))
    train_dl = torch.utils.data.DataLoader(ds, batch_size=2)
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=2, limit_train_batches=5)
    trainer.finetune(model, train_dl, strategy=("freeze_unfreeze", 1))
    image, label = ds[0][DataKeys.INPUT], ds[0][DataKeys.TARGET]
    predictions = model.predict([{DataKeys.INPUT: image}])
    assert (torch.tensor(predictions) > 1).sum() == 0
    assert (torch.tensor(predictions) < 0).sum() == 0
    assert len(predictions[0]) == num_classes == len(label)
    assert len(torch.unique(label)) <= 2


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
@pytest.mark.parametrize("jitter, args", [(torch.jit.script, ()), (torch.jit.trace, (torch.rand(1, 3, 32, 32),))])
def test_jit(tmpdir, jitter, args):
    path = os.path.join(tmpdir, "test.pt")

    model = ImageClassifier(2)
    model.eval()

    model = jitter(model, *args)

    torch.jit.save(model, path)
    model = torch.jit.load(path)

    out = model(torch.rand(1, 3, 32, 32))
    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([1, 2])


@pytest.mark.skipif(not _SERVE_TESTING, reason="serve libraries aren't installed.")
@mock.patch("flash._IS_TESTING", True)
def test_serve():
    model = ImageClassifier(2)
    # TODO: Currently only servable once a input_transform has been attached
    model._input_transform = ImageClassificationInputTransform()
    model.eval()
    model.serve()


@pytest.mark.skipif(_IMAGE_AVAILABLE, reason="image libraries are installed.")
def test_load_from_checkpoint_dependency_error():
    with pytest.raises(ModuleNotFoundError, match=re.escape("'lightning-flash[image]'")):
        ImageClassifier.load_from_checkpoint("not_a_real_checkpoint.pt")


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
def test_cli():
    cli_args = ["flash", "image_classification", "--trainer.fast_dev_run", "True"]
    with mock.patch("sys.argv", cli_args):
        try:
            main()
        except SystemExit:
            pass
