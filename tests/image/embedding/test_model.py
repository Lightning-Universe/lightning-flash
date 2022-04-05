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

import pytest
import torch

import flash
from flash.core.utilities.imports import (
    _IMAGE_AVAILABLE,
    _PL_GREATER_EQUAL_1_5_0,
    _TORCHVISION_AVAILABLE,
    _VISSL_AVAILABLE,
)
from flash.image import ImageClassificationData, ImageEmbedder
from tests.helpers.task_tester import TaskTester

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets import FakeData
else:
    FakeData = object


class TestImageEmbedder(TaskTester):

    task = ImageEmbedder
    task_kwargs = dict(
        backbone="vision_transformer",
        training_strategy="simclr",
        head="simclr_head",
        pretraining_transform="simclr_transform",
    )
    is_testing = is_available = _IMAGE_AVAILABLE and _VISSL_AVAILABLE

    # TODO: Figure out why VISSL can't be jitted
    traceable = False
    scriptable = False

    @property
    def example_forward_input(self):
        return torch.rand(1, 3, 64, 64)

    def check_forward_output(self, output: Any):
        assert isinstance(output, torch.Tensor)
        assert output.shape == torch.Size([1, 384])


@pytest.mark.skipif(torch.cuda.device_count() > 1, reason="VISSL integration doesn't support multi-GPU")
@pytest.mark.skipif(not (_IMAGE_AVAILABLE and _VISSL_AVAILABLE), reason="vissl not installed.")
@pytest.mark.parametrize(
    "backbone, training_strategy, head, pretraining_transform",
    [
        ("vision_transformer", "simclr", "simclr_head", "simclr_transform"),
        pytest.param(
            "vision_transformer",
            "dino",
            "dino_head",
            "dino_transform",
            marks=pytest.mark.skipif(torch.cuda.device_count() < 1, reason="VISSL DINO calls all_reduce internally."),
        ),
        ("vision_transformer", "barlow_twins", "barlow_twins_head", "barlow_twins_transform"),
        ("vision_transformer", "swav", "swav_head", "swav_transform"),
    ],
)
def test_vissl_training(backbone, training_strategy, head, pretraining_transform):
    # moco strategy, transform and head is not added for this test as it doesn't work as of now.
    datamodule = ImageClassificationData.from_datasets(
        train_dataset=FakeData(16),
        predict_dataset=FakeData(8),
        batch_size=4,
    )

    embedder = ImageEmbedder(
        backbone=backbone,
        training_strategy=training_strategy,
        head=head,
        pretraining_transform=pretraining_transform,
    )

    kwargs = {}

    # DINO only works with DDP
    if training_strategy == "dino":
        if _PL_GREATER_EQUAL_1_5_0:
            kwargs["strategy"] = "ddp"
        else:
            kwargs["accelerator"] = "ddp"

    trainer = flash.Trainer(
        max_steps=3,
        max_epochs=1,
        gpus=torch.cuda.device_count(),
        **kwargs,
    )

    trainer.fit(embedder, datamodule=datamodule)
    predictions = trainer.predict(embedder, datamodule=datamodule)
    for prediction_batch in predictions:
        for prediction in prediction_batch:
            assert prediction.size(0) == 384
