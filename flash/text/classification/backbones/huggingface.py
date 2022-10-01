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
#
#
# ResNet encoder adapted from: https://github.com/facebookresearch/swav/blob/master/src/resnet50.py
# as the official torchvision implementation does not support wide resnet architecture
# found in self-supervised learning model weights
from flash.core.registry import ExternalRegistry, FlashRegistry
from flash.core.utilities.imports import _TRANSFORMERS_AVAILABLE
from flash.core.utilities.providers import _HUGGINGFACE
from flash.text.classification.adapters import HuggingFaceAdapter

if _TRANSFORMERS_AVAILABLE:
    from transformers import AutoModelForSequenceClassification


def load_hugingface(backbone: str, num_classes: int):
    model = AutoModelForSequenceClassification.from_pretrained(backbone, num_labels=num_classes)
    return model, backbone


HUGGINGFACE_BACKBONES = FlashRegistry("backbones")

if _TRANSFORMERS_AVAILABLE:

    HUGGINGFACE_BACKBONES = ExternalRegistry(
        getter=load_hugingface,
        name="backbones",
        providers=_HUGGINGFACE,
        adapter=HuggingFaceAdapter,
    )
