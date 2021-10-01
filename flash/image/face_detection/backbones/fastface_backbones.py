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
from functools import partial

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _FASTFACE_AVAILABLE

if _FASTFACE_AVAILABLE:
    import fastface as ff

    _MODEL_NAMES = ff.list_pretrained_models()
else:
    _MODEL_NAMES = []


def fastface_backbone(model_name: str, pretrained: bool, **kwargs):
    if pretrained:
        pl_model = ff.FaceDetector.from_pretrained(model_name, **kwargs)
    else:
        arch, config = model_name.split("_")
        pl_model = ff.FaceDetector.build(arch, config, **kwargs)

    backbone = getattr(pl_model, "arch")

    return backbone, pl_model


def register_ff_backbones(register: FlashRegistry):
    if _FASTFACE_AVAILABLE:
        backbones = [partial(fastface_backbone, model_name=name) for name in _MODEL_NAMES]

        for idx, backbone in enumerate(backbones):
            register(backbone, name=_MODEL_NAMES[idx])
