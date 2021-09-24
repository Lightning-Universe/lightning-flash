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

if _TRANSFORMERS_AVAILABLE:
    from transformers import AutoModelForSequenceClassification

TEXT_CLASSIFIER_BACKBONES = FlashRegistry("backbones")

if _TRANSFORMERS_AVAILABLE:
    HUGGINGFACE_TEXT_CLASSIFIER_BACKBONES = ExternalRegistry(
        getter=AutoModelForSequenceClassification.from_pretrained,
        name="backbones",
        providers=_HUGGINGFACE,
    )
    TEXT_CLASSIFIER_BACKBONES += HUGGINGFACE_TEXT_CLASSIFIER_BACKBONES
