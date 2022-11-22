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

import numpy as np
import pytest
import torch
from torch import Tensor

from flash import Trainer
from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _CORE_TESTING, _SKLEARN_AVAILABLE
from flash.template import TemplateSKLearnClassifier
from flash.template.classification.data import TemplateData

if _SKLEARN_AVAILABLE:
    from sklearn import datasets

# ======== Mock functions ========


class DummyDataset(torch.utils.data.Dataset):
    """We create one or more ``DummyDataset`` classes to provide random data to the model for testing."""

    num_classes: int = 3
    num_features: int = 4

    def __getitem__(self, index):
        return {
            DataKeys.INPUT: torch.randn(self.num_features),
            DataKeys.TARGET: torch.randint(self.num_classes - 1, (1,))[0],
        }

    def __len__(self) -> int:
        return 10


# ==============================


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_smoke():
    """A simple test that the class can be instantiated."""
    model = TemplateSKLearnClassifier(num_features=1, num_classes=1)
    assert model is not None


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
@pytest.mark.parametrize("num_classes", [4, 256])
@pytest.mark.parametrize("shape", [(1, 3), (2, 128)])
def test_forward(num_classes, shape):
    """Tests that a tensor can be given to the model forward and gives the correct output size."""
    model = TemplateSKLearnClassifier(
        num_features=shape[1],
        num_classes=num_classes,
    )
    model.eval()

    row = torch.rand(*shape)

    out = model(row)
    assert out.shape == (shape[0], num_classes)


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_train(tmpdir):
    """Tests that the model can be trained on our ``DummyDataset``."""
    model = TemplateSKLearnClassifier(num_features=DummyDataset.num_features, num_classes=DummyDataset.num_classes)
    train_dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=4)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, train_dl)


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_val(tmpdir):
    """Tests that the model can be validated on our ``DummyDataset``."""
    model = TemplateSKLearnClassifier(num_features=DummyDataset.num_features, num_classes=DummyDataset.num_classes)
    val_dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=4)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.validate(model, val_dl)


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_test(tmpdir):
    """Tests that the model can be tested on our ``DummyDataset``."""
    model = TemplateSKLearnClassifier(num_features=DummyDataset.num_features, num_classes=DummyDataset.num_classes)
    test_dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=4)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.test(model, test_dl)


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_predict_numpy():
    """Tests that we can generate predictions from a numpy array."""
    row = np.random.rand(1, DummyDataset.num_features)
    model = TemplateSKLearnClassifier(num_features=DummyDataset.num_features, num_classes=DummyDataset.num_classes)
    datamodule = TemplateData.from_numpy(predict_data=row, batch_size=1)
    trainer = Trainer()
    out = trainer.predict(model, datamodule=datamodule, output="classes")
    assert isinstance(out[0][0], int)


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_predict_sklearn():
    """Tests that we can generate predictions from a scikit-learn ``Bunch``."""
    bunch = datasets.load_iris()
    model = TemplateSKLearnClassifier(num_features=DummyDataset.num_features, num_classes=DummyDataset.num_classes)
    datamodule = TemplateData.from_sklearn(predict_bunch=bunch, batch_size=1)
    trainer = Trainer()
    out = trainer.predict(model, datamodule=datamodule, output="classes")
    assert isinstance(out[0][0], int)


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
@pytest.mark.parametrize("jitter, args", [(torch.jit.script, ()), (torch.jit.trace, (torch.rand(1, 16),))])
def test_jit(tmpdir, jitter, args):
    path = os.path.join(tmpdir, "testing_model.pt")

    model = TemplateSKLearnClassifier(num_features=16, num_classes=10)
    model.eval()

    model = jitter(model, *args)

    torch.jit.save(model, path)
    model = torch.jit.load(path)

    out = model(torch.rand(1, 16))
    assert isinstance(out, Tensor)
    assert out.shape == torch.Size([1, 10])
