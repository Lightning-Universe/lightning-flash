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
from pytorch_lightning.utilities import _BOLTS_AVAILABLE, _TORCHVISION_AVAILABLE

from flash.image.segmentation.backbones import SEMANTIC_SEGMENTATION_BACKBONES


@pytest.mark.parametrize(["backbone"], [
    pytest.param("fcn_resnet50", marks=pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="No torchvision")),
    pytest.param("deeplabv3_resnet50", marks=pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="No torchvision")),
    pytest.param(
        "lraspp_mobilenet_v3_large", marks=pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="No torchvision")
    ),
    pytest.param("unet", marks=pytest.mark.skipif(not _BOLTS_AVAILABLE, reason="No bolts")),
])
def test_image_classifier_backbones_registry(backbone):
    img = torch.rand(1, 3, 32, 32)
    backbone_fn = SEMANTIC_SEGMENTATION_BACKBONES.get(backbone)
    backbone_model = backbone_fn(10, pretrained=False)
    assert backbone_model
    backbone_model.eval()
    res = backbone_model(img)
    if isinstance(res, dict):
        res = res["out"]
    assert res.shape[1] == 10
