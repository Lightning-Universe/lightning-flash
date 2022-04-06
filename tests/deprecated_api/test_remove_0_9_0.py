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

from flash.core.utilities.imports import _VISSL_AVAILABLE
from flash.image.embedding.model import ImageEmbedder


@pytest.mark.skipif(not _VISSL_AVAILABLE, reason="vissl not installed.")
@pytest.mark.parametrize(
    "deprecated_backbone, alternative_backbone",
    [("resnet", "resnet50"), ("vision_transformer", "vit_small_patch16_224")],
)
def test_0_9_0_embedder_models(deprecated_backbone, alternative_backbone):
    with pytest.warns(FutureWarning, match=f"Use '{alternative_backbone}' instead."):
        ImageEmbedder(
            backbone=deprecated_backbone,
            training_strategy="simclr",
            head="simclr_head",
            pretraining_transform="simclr_transform",
        )
