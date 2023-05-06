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

import numpy as np
import pytest
import torch
from torch import Tensor

from flash import Trainer
from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _SEGMENTATION_MODELS_AVAILABLE, _TOPIC_IMAGE_AVAILABLE, _TOPIC_SERVE_AVAILABLE
from flash.image import SemanticSegmentation
from flash.image.segmentation.data import SemanticSegmentationData
from tests.helpers.task_tester import TaskTester


@pytest.mark.skipif(not _SEGMENTATION_MODELS_AVAILABLE, reason="No SMP")
class TestSemanticSegmentation(TaskTester):
    task = SemanticSegmentation
    task_args = (2,)
    cli_command = "semantic_segmentation"
    is_testing = _TOPIC_IMAGE_AVAILABLE
    is_available = _TOPIC_IMAGE_AVAILABLE
    scriptable = False

    @property
    def example_forward_input(self):
        return torch.rand(1, 3, 32, 32)

    def check_forward_output(self, output: Any):
        assert isinstance(output, Tensor)
        assert output.shape == torch.Size([1, 2, 32, 32])

    @property
    def example_train_sample(self):
        return {
            DataKeys.INPUT: torch.rand(3, 224, 224),
            DataKeys.TARGET: torch.randint(2, (224, 224)),
        }

    @property
    def example_val_sample(self):
        return self.example_train_sample

    @property
    def example_test_sample(self):
        return self.example_train_sample


@pytest.mark.skipif(not _SEGMENTATION_MODELS_AVAILABLE, reason="No SMP")
def test_non_existent_backbone():
    with pytest.raises(KeyError):
        SemanticSegmentation(2, "i am never going to implement this lol")


@pytest.mark.skipif(not _SEGMENTATION_MODELS_AVAILABLE, reason="No SMP")
def test_freeze():
    model = SemanticSegmentation(2)
    model.freeze()
    for p in model.backbone.parameters():
        assert p.requires_grad is False


@pytest.mark.skipif(not _SEGMENTATION_MODELS_AVAILABLE, reason="No SMP")
def test_unfreeze():
    model = SemanticSegmentation(2)
    model.unfreeze()
    for p in model.backbone.parameters():
        assert p.requires_grad is True


@pytest.mark.skipif(not _SEGMENTATION_MODELS_AVAILABLE, reason="No SMP")
def test_predict_tensor():
    img = torch.rand(1, 3, 64, 64)
    model = SemanticSegmentation(2, backbone="mobilenetv3_large_100")
    datamodule = SemanticSegmentationData.from_tensors(predict_data=img, batch_size=1)
    trainer = Trainer()
    out = trainer.predict(model, datamodule=datamodule, output="labels")
    assert isinstance(out[0][0], list)
    assert len(out[0][0]) == 64
    assert len(out[0][0][0]) == 64


@pytest.mark.skipif(not _SEGMENTATION_MODELS_AVAILABLE, reason="No SMP")
def test_predict_numpy():
    img = np.ones((1, 3, 64, 64))
    model = SemanticSegmentation(2, backbone="mobilenetv3_large_100")
    datamodule = SemanticSegmentationData.from_numpy(predict_data=img, batch_size=1)
    trainer = Trainer()
    out = trainer.predict(model, datamodule=datamodule, output="labels")
    assert isinstance(out[0][0], list)
    assert len(out[0][0]) == 64
    assert len(out[0][0][0]) == 64


@pytest.mark.skipif(not _SEGMENTATION_MODELS_AVAILABLE, reason="No SMP")
@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="some serving")
@mock.patch("flash._IS_TESTING", True)
def test_serve():
    model = SemanticSegmentation(2)
    model.eval()
    model.serve()


@pytest.mark.skipif(not _SEGMENTATION_MODELS_AVAILABLE, reason="No SMP")
def test_available_pretrained_weights():
    assert SemanticSegmentation.available_pretrained_weights("resnet18") == ["imagenet", "ssl", "swsl"]
