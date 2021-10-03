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
import torch
import transformers
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from typing import Dict, Tuple

from flash.core.registry import FlashRegistry


def _forward_avg(x: BaseModelOutputWithPastAndCrossAttentions) -> torch.Tensor:
    return x.last_hidden_state.mean(dim=1)


def _forward_cls(x: BaseModelOutputWithPoolingAndCrossAttentions) -> torch.Tensor:
    return x.last_hidden_state[:, 0, :]


def _forward_pooler(x: BaseModelOutputWithPoolingAndCrossAttentions) -> torch.Tensor:
    return x.pooler_output


AVAILABLE_STRATEGIES: FlashRegistry = FlashRegistry("strategies")

AVAILABLE_STRATEGIES(
    fn=_forward_avg,
    name="avg",
)

AVAILABLE_STRATEGIES(
    fn=_forward_cls,
    name="cls_token",
)

AVAILABLE_STRATEGIES(
    fn=_forward_pooler,
    name="pooler_output",
)


class Transformer(nn.Module):

    available_strategies: FlashRegistry = AVAILABLE_STRATEGIES

    def __init__(self, model_name: str = "prajjwal1/bert-tiny", pretrained: bool = True, strategy: str = "cls_token"):
        super().__init__()
        self.model_name = model_name
        self.strategy = strategy
        self.config = AutoConfig.from_pretrained(model_name)
        self.vocab_size = self.config.vocab_size

        if pretrained:
            self.model = AutoModel.from_pretrained(self.model_name)
        else:
            self.model = AutoModel.from_config(self.config)

        self._sentence_representation = AVAILABLE_STRATEGIES.get(strategy)

    def forward(self, x: Dict[int, torch.Tensor]) -> torch.Tensor:
        x = self.model(x["input_ids"], x["attention_mask"])
        return self._sentence_representation(x)


def _trasformer(
    model_name: str = "prajjwal1/bert-tiny",
    pretrained: bool = True,
    strategy: str = "cls_token",
) -> Tuple[nn.Module, int]:
    # disable HF thousand warnings when loading model
    transformers.logging.set_verbosity_error()

    model = Transformer(model_name, pretrained, strategy)

    # re-enable HF warnings
    transformers.logging.set_verbosity_warning()

    return model, model.config.hidden_size
