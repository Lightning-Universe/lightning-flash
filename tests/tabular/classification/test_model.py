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
from typing import Any
from unittest import mock

import pandas as pd
import pytest
import torch
from pytorch_lightning import Trainer

from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _SERVE_TESTING, _TABULAR_AVAILABLE, _TABULAR_TESTING
from flash.tabular.classification.data import TabularClassificationData
from flash.tabular.classification.model import TabularClassifier
from tests.helpers.task_tester import TaskTester

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
        return {DataKeys.INPUT: (cat_vars, num_vars), DataKeys.TARGET: target}

    def __len__(self) -> int:
        return 100


# ==============================


class TestTabularClassifier(TaskTester):

    task = TabularClassifier
    task_kwargs = {
        "parameters": {"categorical_fields": list(range(4))},
        "embedding_sizes": [(10, 32) for _ in range(4)],
        "cat_dims": [10 for _ in range(4)],
        "num_features": 8,
        "num_classes": 10,
        "backbone": "tabnet",
    }
    cli_command = "tabular_classification"
    is_testing = _TABULAR_TESTING
    is_available = _TABULAR_AVAILABLE

    # TODO: Resolve JIT issues
    scriptable = False
    traceable = False

    @property
    def example_forward_input(self):
        return {
            "continuous": torch.rand(1, 4),
            "categorical": torch.randint(0, 10, size=(1, 4)),
        }

    def check_forward_output(self, output: Any):
        assert isinstance(output, torch.Tensor)
        assert output.shape == torch.Size([1, 10])


@pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular libraries aren't installed.")
@pytest.mark.parametrize(
    "backbone", ["tabnet", "tabtransformer", "fttransformer", "autoint", "node", "category_embedding"]
)
def test_init_train(backbone, tmpdir):
    train_dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=16)
    data_properties = {
        "parameters": {"categorical_fields": list(range(16))},
        "embedding_sizes": [(10, 32) for _ in range(16)],
        "cat_dims": [10 for _ in range(16)],
        "num_features": 32,
        "num_classes": 10,
        "backbone": backbone,
    }

    model = TabularClassifier(**data_properties)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, train_dl)


@pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular libraries aren't installed.")
@pytest.mark.parametrize(
    "backbone", ["tabnet", "tabtransformer", "fttransformer", "autoint", "node", "category_embedding"]
)
def test_init_train_no_num(backbone, tmpdir):
    train_dl = torch.utils.data.DataLoader(DummyDataset(num_num=0), batch_size=16)
    data_properties = {
        "parameters": {"categorical_fields": list(range(16))},
        "embedding_sizes": [(10, 32) for _ in range(16)],
        "cat_dims": [10 for _ in range(16)],
        "num_features": 16,
        "num_classes": 10,
        "backbone": backbone,
    }

    model = TabularClassifier(**data_properties)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, train_dl)


@pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular libraries aren't installed.")
@pytest.mark.parametrize("backbone", ["tabnet", "tabtransformer", "autoint", "node", "category_embedding"])
def test_init_train_no_cat(backbone, tmpdir):
    train_dl = torch.utils.data.DataLoader(DummyDataset(num_cat=0), batch_size=16)
    data_properties = {
        "parameters": {"categorical_fields": []},
        "embedding_sizes": [],
        "cat_dims": [],
        "num_features": 16,
        "num_classes": 10,
        "backbone": backbone,
    }

    model = TabularClassifier(**data_properties)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, train_dl)


@pytest.mark.skipif(not _SERVE_TESTING, reason="serve libraries aren't installed.")
@pytest.mark.parametrize(
    "backbone", ["tabnet", "tabtransformer", "fttransformer", "autoint", "node", "category_embedding"]
)
@mock.patch("flash._IS_TESTING", True)
def test_serve(backbone):
    train_data = {"num_col": [1.4, 2.5], "cat_col": ["positive", "negative"], "target": [1, 2]}
    datamodule = TabularClassificationData.from_data_frame(
        "cat_col",
        "num_col",
        "target",
        train_data_frame=pd.DataFrame.from_dict(train_data),
        batch_size=1,
    )
    model = TabularClassifier.from_data(datamodule=datamodule, backbone=backbone)
    model.eval()
    model.serve(parameters=datamodule.parameters)
