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
import pytest
import torch

from flash.core.utilities.imports import _BOLTS_AVAILABLE, _TORCHVISION_AVAILABLE
from flash.image.segmentation.backbones import SEMANTIC_SEGMENTATION_BACKBONES
from flash.image.segmentation.heads import SEMANTIC_SEGMENTATION_HEADS


@pytest.mark.parametrize(
    "head", [
        pytest.param("fcn", marks=pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="No torchvision")),
        pytest.param("deeplabv3", marks=pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="No torchvision")),
        pytest.param("lraspp", marks=pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="No torchvision")),
        pytest.param("unet", marks=pytest.mark.skipif(not _BOLTS_AVAILABLE, reason="No bolts")),
    ]
)
def test_semantic_segmentation_heads_registry(head):
    img = torch.rand(1, 3, 32, 32)
    backbone = SEMANTIC_SEGMENTATION_BACKBONES.get("resnet50")(pretrained=False)
    head = SEMANTIC_SEGMENTATION_HEADS.get(head)(backbone, 10)
    assert backbone
    assert head
    head.eval()
    res = head(img)
    if isinstance(res, dict):
        res = res["out"]
    assert res.shape[1] == 10
