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

from flash.core.utilities.imports import _SEGMENTATION_MODELS_AVAILABLE
from flash.image.segmentation.backbones import SEMANTIC_SEGMENTATION_BACKBONES


@pytest.mark.parametrize("backbone", ["resnet50", "dpn131"])
@pytest.mark.skipif(not _SEGMENTATION_MODELS_AVAILABLE, reason="No SMP")
def test_semantic_segmentation_backbones_registry(backbone):
    backbone = SEMANTIC_SEGMENTATION_BACKBONES.get(backbone)()
    assert backbone
    assert isinstance(backbone, str)
