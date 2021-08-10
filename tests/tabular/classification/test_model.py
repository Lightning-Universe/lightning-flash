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
import re
from unittest import mock

import pandas as pd
import pytest
import torch
from pytorch_lightning import Trainer

from flash.core.data.data_source import DefaultDataKeys
from flash.core.utilities.imports import _TABULAR_AVAILABLE
from flash.tabular.classification.data import TabularClassificationData
from flash.tabular.classification.model import TabularClassifier
from tests.helpers.utils import _SERVE_TESTING, _TABULAR_TESTING

# ======== Mock functions ========


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_num=16, num_cat=16):
        super().__init__()
        self.num_num = num_num
        self.num_cat = num_cat

    def __getitem__(self, index):
        target = torch.randint(0, 10, size=(1,)).item()
        cat_vars = torch.randint(0, 10, size=(self.num_cat,))
        num_vars = torch.rand(self.num_num)
        return {DefaultDataKeys.INPUT: (cat_vars, num_vars), DefaultDataKeys.TARGET: target}

    def __len__(self) -> int:
        return 100


# ==============================


@pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular libraries aren't installed.")
def test_init_train(tmpdir):
    train_dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=16)
    model = TabularClassifier(num_classes=10, num_features=16 + 16, embedding_sizes=16 * [(10, 32)])
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, train_dl)


@pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular libraries aren't installed.")
def test_init_train_no_num(tmpdir):
    train_dl = torch.utils.data.DataLoader(DummyDataset(num_num=0), batch_size=16)
    model = TabularClassifier(num_classes=10, num_features=16, embedding_sizes=16 * [(10, 32)])
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, train_dl)


@pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular libraries aren't installed.")
def test_init_train_no_cat(tmpdir):
    train_dl = torch.utils.data.DataLoader(DummyDataset(num_cat=0), batch_size=16)
    model = TabularClassifier(num_classes=10, num_features=16, embedding_sizes=[])
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, train_dl)


@pytest.mark.skipif(_TABULAR_AVAILABLE, reason="tabular libraries are installed.")
def test_module_import_error(tmpdir):
    with pytest.raises(ModuleNotFoundError, match="[tabular]"):
        TabularClassifier(num_classes=10, num_features=16, embedding_sizes=[])


@pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular libraries aren't installed.")
def test_jit(tmpdir):
    model = TabularClassifier(num_classes=10, num_features=8, embedding_sizes=4 * [(10, 32)])
    model.eval()

    # torch.jit.script doesn't work with tabnet
    model = torch.jit.trace(model, ((torch.randint(0, 10, size=(1, 4)), torch.rand(1, 4)),))

    # TODO: torch.jit.save doesn't work with tabnet
    # path = os.path.join(tmpdir, "test.pt")
    # torch.jit.save(model, path)
    # model = torch.jit.load(path)

    out = model((torch.randint(0, 10, size=(1, 4)), torch.rand(1, 4)))
    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([1, 10])


@pytest.mark.skipif(not _SERVE_TESTING, reason="serve libraries aren't installed.")
@mock.patch("flash._IS_TESTING", True)
def test_serve():
    train_data = {"num_col": [1.4, 2.5], "cat_col": ["positive", "negative"], "target": [1, 2]}
    datamodule = TabularClassificationData.from_data_frame(
        "cat_col",
        "num_col",
        "target",
        pd.DataFrame.from_dict(train_data),
    )
    model = TabularClassifier.from_data(datamodule)
    # TODO: Currently only servable once a preprocess has been attached
    model._preprocess = datamodule.preprocess
    model.eval()
    model.serve()


@pytest.mark.skipif(_TABULAR_AVAILABLE, reason="tabular libraries are installed.")
def test_load_from_checkpoint_dependency_error():
    with pytest.raises(ModuleNotFoundError, match=re.escape("'lightning-flash[tabular]'")):
        TabularClassifier.load_from_checkpoint("not_a_real_checkpoint.pt")
