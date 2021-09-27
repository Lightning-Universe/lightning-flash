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
import unittest.mock

import pytest
import torch

from flash.core.utilities.imports import _SEGMENTATION_MODELS_AVAILABLE
from flash.image.segmentation import SemanticSegmentation
from flash.image.segmentation.backbones import SEMANTIC_SEGMENTATION_BACKBONES
from flash.image.segmentation.heads import SEMANTIC_SEGMENTATION_HEADS
from tests.helpers.utils import _IMAGE_TESTING


@pytest.mark.parametrize(
    "head",
    [
        pytest.param("fpn", marks=pytest.mark.skipif(not _SEGMENTATION_MODELS_AVAILABLE, reason="No SMP")),
        pytest.param("deeplabv3", marks=pytest.mark.skipif(not _SEGMENTATION_MODELS_AVAILABLE, reason="No SMP")),
        pytest.param("unet", marks=pytest.mark.skipif(not _SEGMENTATION_MODELS_AVAILABLE, reason="No SMP")),
    ],
)
def test_semantic_segmentation_heads_registry(head):
    img = torch.rand(1, 3, 32, 32)
    backbone = SEMANTIC_SEGMENTATION_BACKBONES.get("resnet50")(pretrained=False)
    head = SEMANTIC_SEGMENTATION_HEADS.get(head)(backbone=backbone, num_classes=10)
    assert backbone
    assert head
    head.eval()
    res = head(img)
    if isinstance(res, dict):
        res = res["out"]
    assert res.shape[1] == 10


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
@unittest.mock.patch("flash.image.segmentation.heads.smp")
def test_pretrained_weights(mock_smp):
    mock_smp.create_model = unittest.mock.MagicMock()
    available_weights = SemanticSegmentation.available_pretrained_weights("resnet18")
    backbone = SEMANTIC_SEGMENTATION_BACKBONES.get("resnet18")()
    SEMANTIC_SEGMENTATION_HEADS.get("unet")(backbone=backbone, num_classes=10, pretrained=True)

    kwargs = {
        "arch": "unet",
        "classes": 10,
        "encoder_name": "resnet18",
        "in_channels": 3,
        "encoder_weights": "imagenet",
    }
    mock_smp.create_model.assert_called_with(**kwargs)

    for weight in available_weights:
        SEMANTIC_SEGMENTATION_HEADS.get("unet")(backbone=backbone, num_classes=10, pretrained=weight)
        kwargs["encoder_weights"] = weight
        mock_smp.create_model.assert_called_with(**kwargs)
