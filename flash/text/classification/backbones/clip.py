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

import torch
from torch import nn

from flash.core.registry import FlashRegistry
from flash.core.utilities.providers import _CLIP
from flash.core.utilities.url_error import catch_url_error
from flash.text.classification.adapters import GenericAdapter

# Paper: Learning Transferable Visual Models From Natural Language Supervision
# https://arxiv.org/abs/2103.00020 from Alec Radford et. al. (26 Feb 2021)
# weights from https://github.com/openai/CLIP


_CLIP_MODELS = {
    "RN50": "resnet50",
    "RN101": "resnet101",
    "RN50x4": "resrnet50x4",
    "RN50x16": "resrnet50x16",
    "RN50x64": "resrnet50x64",
    "ViT_B_32": "vitb32",
    "ViT_B_16": "vitb16",
    "ViT_L_14": "vitl14",
    "ViT_L_14_336px": "vitl14@336px",
}


class _CLIPWrapper(nn.Module):
    def __init__(self, clip_model: nn.Module):
        super().__init__()

        self.clip_model = clip_model

    def forward(self, x):
        return self.clip_model.encode_text(x)


def _load_clip(model_name: str, **kwargs):
    backbone, _ = torch.hub.load("openai/CLIP:main", model_name)
    tokenizer = torch.hub.load("openai/CLIP:main", "tokenize")
    tokenizer = partial(tokenizer, truncate=True)
    return _CLIPWrapper(backbone), tokenizer, backbone.visual.output_dim


CLIP_BACKBONES = FlashRegistry("backbones")

for clip_model_name, flash_model_name in _CLIP_MODELS.items():
    CLIP_BACKBONES(
        catch_url_error(partial(_load_clip, clip_model_name)),
        f"clip_{flash_model_name}",
        providers=_CLIP,
        adapter=GenericAdapter,
    )
