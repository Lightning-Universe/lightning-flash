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
from functools import partial
from typing import Union

from huggingface_hub import HfApi
from transformers import AutoModelForSequenceClassification

from flash.core.registry import ConcatRegistry, ExternalRegistry, FlashRegistry
from flash.core.utilities.imports import _TRANSFORMERS_AVAILABLE
from flash.core.utilities.providers import _HUGGINGFACE
from flash.core.utilities.url_error import catch_url_error


def register_trasformers_backbones() -> Union[FlashRegistry, ConcatRegistry]:

    register = FlashRegistry("backbones")

    try:

        model_list = HfApi().list_models(filter=("pytorch", "text-classification"))

        for model_name in map(lambda x: x.modelId, model_list):
            register(
                fn=catch_url_error(partial(AutoModelForSequenceClassification.from_pretrained, model_name)),
                name=model_name,
                providers=_HUGGINGFACE,
            )

    except:

        register = register + ExternalRegistry(
            getter=AutoModelForSequenceClassification.from_pretrained,
            name="backbones",
            providers=_HUGGINGFACE,
        )

    return register


TEXT_CLASSIFIER_BACKBONES = FlashRegistry("backbones")

if _TRANSFORMERS_AVAILABLE:
    TEXT_CLASSIFIER_BACKBONES += register_trasformers_backbones()


print(TEXT_CLASSIFIER_BACKBONES)