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
from unittest.mock import patch

import flash
import pytest
import torch
from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _TOPIC_SERVE_AVAILABLE, _TOPIC_TEXT_AVAILABLE, _TORCH_ORT_AVAILABLE
from flash.text import TextClassifier
from flash.text.ort_callback import ORTCallback
from torch import Tensor

from tests.helpers.boring_model import BoringModel
from tests.helpers.task_tester import StaticDataset, TaskTester

TEST_BACKBONE = "prajjwal1/bert-tiny"  # tiny model for testing


class TestTextClassifier(TaskTester):
    task = TextClassifier
    task_args = (2,)
    task_kwargs = {"backbone": TEST_BACKBONE}
    cli_command = "text_classification"
    is_testing = _TOPIC_TEXT_AVAILABLE
    is_available = _TOPIC_TEXT_AVAILABLE

    scriptable = False

    marks = {
        "test_fit": [
            pytest.mark.parametrize(
                "task_kwargs",
                [
                    {},
                    pytest.param(
                        {"enable_ort": True},
                        marks=pytest.mark.skipif(not _TORCH_ORT_AVAILABLE, reason="ORT Module aren't installed."),
                    ),
                    {"backbone": "clip_resnet50"},
                ],
            )
        ],
        "test_val": [
            pytest.mark.parametrize(
                "task_kwargs",
                [
                    {},
                    {"backbone": "clip_resnet50"},
                ],
            )
        ],
        "test_test": [
            pytest.mark.parametrize(
                "task_kwargs",
                [
                    {},
                    {"backbone": "clip_resnet50"},
                ],
            )
        ],
        "test_cli": [pytest.mark.parametrize("extra_args", ([], ["from_toxic"]))],
    }

    @property
    def example_forward_input(self):
        return {"input_ids": torch.randint(1000, size=(1, 100))}

    def check_forward_output(self, output: Any):
        assert isinstance(output, Tensor)
        assert output.shape == torch.Size([1, 2])

    @property
    def example_train_sample(self):
        return {DataKeys.INPUT: "some text", DataKeys.TARGET: 1}

    @property
    def example_val_sample(self):
        return self.example_train_sample

    @property
    def example_test_sample(self):
        return self.example_train_sample

    @pytest.mark.skipif(not _TORCH_ORT_AVAILABLE, reason="ORT Module aren't installed.")
    def test_ort_callback_fails_no_model(self, tmpdir):
        dataset = StaticDataset(self.example_train_sample, 4)

        model = BoringModel()

        trainer = flash.Trainer(default_root_dir=tmpdir, fast_dev_run=True, callbacks=ORTCallback())

        with pytest.raises(ValueError, match="Torch ORT requires to wrap a single model"):
            trainer.fit(model, model.process_train_dataset(dataset, batch_size=4))


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="serve libraries aren't installed.")
@patch("flash._IS_TESTING", True)
def test_serve():
    model = TextClassifier(2, backbone=TEST_BACKBONE)
    model.eval()
    model.serve()
