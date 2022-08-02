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
from torch import Tensor

import flash
from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _SERVE_TESTING, _TABULAR_AVAILABLE, _TABULAR_TESTING
from flash.tabular.classification.data import TabularClassificationData
from flash.tabular.classification.model import TabularClassifier
from tests.helpers.task_tester import StaticDataset, TaskTester


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

    marks = {
        "test_fit": [
            pytest.mark.parametrize(
                "task_kwargs",
                [
                    {"backbone": "tabnet"},
                    {"backbone": "tabtransformer"},
                    {"backbone": "fttransformer"},
                    {"backbone": "autoint"},
                    {"backbone": "node"},
                    {"backbone": "category_embedding"},
                ],
            )
        ],
        "test_val": [
            pytest.mark.parametrize(
                "task_kwargs",
                [
                    {"backbone": "tabnet"},
                    {"backbone": "tabtransformer"},
                    {"backbone": "fttransformer"},
                    {"backbone": "autoint"},
                    {"backbone": "node"},
                    {"backbone": "category_embedding"},
                ],
            )
        ],
        "test_test": [
            pytest.mark.parametrize(
                "task_kwargs",
                [
                    {"backbone": "tabnet"},
                    {"backbone": "tabtransformer"},
                    {"backbone": "fttransformer"},
                    {"backbone": "autoint"},
                    {"backbone": "node"},
                    {"backbone": "category_embedding"},
                ],
            )
        ],
        "test_cli": [pytest.mark.parametrize("extra_args", ([],))],
    }

    @property
    def example_forward_input(self):
        return {
            "continuous": torch.rand(1, 4),
            "categorical": torch.randint(0, 10, size=(1, 4)),
        }

    def check_forward_output(self, output: Any):
        assert isinstance(output, Tensor)
        assert output.shape == torch.Size([1, 10])

    @property
    def example_train_sample(self):
        return {DataKeys.INPUT: (torch.randint(0, 10, size=(4,)), torch.rand(4)), DataKeys.TARGET: 1}

    @property
    def example_val_sample(self):
        return self.example_train_sample

    @property
    def example_test_sample(self):
        return self.example_train_sample

    @pytest.mark.parametrize(
        "backbone", ["tabnet", "tabtransformer", "fttransformer", "autoint", "node", "category_embedding"]
    )
    def test_init_train_no_num(self, backbone, tmpdir):
        no_num_sample = {DataKeys.INPUT: (torch.randint(0, 10, size=(4,)), torch.empty(0)), DataKeys.TARGET: 1}
        dataset = StaticDataset(no_num_sample, 4)

        args = self.task_args
        kwargs = dict(**self.task_kwargs)
        kwargs.update(num_features=4)
        model = self.task(*args, **kwargs)

        trainer = flash.Trainer(default_root_dir=tmpdir, fast_dev_run=True)
        trainer.fit(model, model.process_train_dataset(dataset, batch_size=4))

    @pytest.mark.parametrize("backbone", ["tabnet", "tabtransformer", "autoint", "node", "category_embedding"])
    def test_init_train_no_cat(self, backbone, tmpdir):
        no_cat_sample = {DataKeys.INPUT: (torch.empty(0), torch.rand(4)), DataKeys.TARGET: 1}
        dataset = StaticDataset(no_cat_sample, 4)

        args = self.task_args
        kwargs = dict(**self.task_kwargs)
        kwargs.update(parameters={"categorical_fields": []}, embedding_sizes=[], cat_dims=[], num_features=4)
        model = self.task(*args, **kwargs)

        trainer = flash.Trainer(default_root_dir=tmpdir, fast_dev_run=True)
        trainer.fit(model, model.process_train_dataset(dataset, batch_size=4))


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
