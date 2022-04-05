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

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE, _VISSL_AVAILABLE
from flash.image.embedding.heads.vissl_heads import SimCLRHead
from flash.image.embedding.vissl.hooks import TrainingSetupHook

if _VISSL_AVAILABLE:
    from vissl.hooks.moco_hooks import MoCoHook
    from vissl.hooks.swav_hooks import NormalizePrototypesHook, SwAVUpdateQueueScoresHook
    from vissl.losses.barlow_twins_loss import BarlowTwinsLoss
    from vissl.losses.moco_loss import MoCoLoss
    from vissl.losses.swav_loss import SwAVLoss
    from vissl.models.heads.swav_prototypes_head import SwAVPrototypesHead

    from flash.image.embedding.strategies import IMAGE_EMBEDDER_STRATEGIES
else:
    MoCoHook = object
    NormalizePrototypesHook = object
    SwAVUpdateQueueScoresHook = object

    BarlowTwinsLoss = object
    MoCoLoss = object
    SwAVLoss = object

    SwAVPrototypesHead = object

    IMAGE_EMBEDDER_STRATEGIES = FlashRegistry("embedder_training_strategies")


@pytest.mark.skipif(not (_TORCHVISION_AVAILABLE and _VISSL_AVAILABLE), reason="vissl not installed.")
@pytest.mark.parametrize(
    "training_strategy, head_name, loss_fn_class, head_class, hooks_list",
    [
        ("barlow_twins", "barlow_twins_head", BarlowTwinsLoss, SimCLRHead, [TrainingSetupHook]),
        ("moco", "moco_head", MoCoLoss, SimCLRHead, [MoCoHook, TrainingSetupHook]),
        (
            "swav",
            "swav_head",
            SwAVLoss,
            SwAVPrototypesHead,
            [SwAVUpdateQueueScoresHook, NormalizePrototypesHook, TrainingSetupHook],
        ),
    ],
)
def test_vissl_strategies(tmpdir, training_strategy, head_name, loss_fn_class, head_class, hooks_list):
    ret_loss_fn, ret_head, ret_hooks = IMAGE_EMBEDDER_STRATEGIES.get(training_strategy)(head=head_name)

    assert isinstance(ret_loss_fn, loss_fn_class)
    assert isinstance(ret_head, head_class)
    for hook in hooks_list:
        hook_present = 0
        for ret_hook in ret_hooks:
            if isinstance(ret_hook, hook):
                hook_present = 1

        assert hook_present == 1
