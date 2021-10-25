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
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    import transformers
    from transformers import AutoConfig, AutoModel
    from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

    def _forward_avg(x: BaseModelOutputWithPoolingAndCrossAttentions) -> torch.Tensor:
        return x.last_hidden_state.mean(dim=1)

    def _forward_cls(x: BaseModelOutputWithPoolingAndCrossAttentions) -> torch.Tensor:
        return x.last_hidden_state[:, 0, :]

    def _forward_pooler(x: BaseModelOutputWithPoolingAndCrossAttentions) -> torch.Tensor:
        return x.pooler_output

    AVAILABLE_STRATEGIES: FlashRegistry = FlashRegistry("strategies")
    AVAILABLE_STRATEGIES(fn=_forward_avg, name="avg")
    AVAILABLE_STRATEGIES(fn=_forward_cls, name="cls_token")
    AVAILABLE_STRATEGIES(fn=_forward_pooler, name="pooler_output")

    class Transformer(nn.Module):

        available_strategies: FlashRegistry = AVAILABLE_STRATEGIES

        def __init__(self, model_name: str, pretrained: bool, strategy: str, vocab_size: Optional[int] = None):
            super().__init__()
            self.model_name = model_name
            self.strategy = strategy
            self.config = AutoConfig.from_pretrained(model_name)
            self.vocab_size = self.config.vocab_size

            if pretrained:
                # load pretrained weights
                self.model = AutoModel.from_pretrained(self.model_name)
            else:
                # initialize model from scratch
                self.model = AutoModel.from_config(self.config)

            if vocab_size:
                # re-initialize the embeddings layer
                self.vocab_size = vocab_size
                print(f"Re-initialize word embeddings layer with `vocab_size={self.vocab_size}`")
                self._init_embeddings()
            else:
                self.vocab_size = self.model.config.vocab_size

            self._sentence_representation = AVAILABLE_STRATEGIES.get(strategy)

        def forward(self, x: Dict[int, torch.Tensor]) -> torch.Tensor:
            x = self.model(x["input_ids"], x["attention_mask"], output_attentions=False, output_hidden_states=False)
            return self._sentence_representation(x)

        def _init_embeddings(self):
            """Re-initializes the embedding layer."""
            num_embeddings = self.model.config.vocab_size
            initializer_range = self.model.config.initializer_range

            for name, module in self.model.named_modules():
                # find the word embedding layer
                if isinstance(module, torch.nn.Embedding) and module.num_embeddings == num_embeddings:
                    embedding_module_name = name
                    embedding_dim = module.embedding_dim
                    padding_idx = module.padding_idx
                    break
            _, name = embedding_module_name.split(".")
            new_embedding_module = torch.nn.Embedding(self.vocab_size, embedding_dim, padding_idx)
            new_embedding_module.weight.data.normal_(mean=0.0, std=initializer_range)

            self.model.embeddings.add_module(name, new_embedding_module)

    def _transformer(
        model_name: str = "prajjwal1/bert-tiny",
        pretrained: bool = True,
        strategy: str = "cls_token",
        vocab_size: Optional[int] = None,
    ) -> Tuple[nn.Module, int]:

        # disable Hugging Face warnings
        transformers.logging.set_verbosity_error()

        model = Transformer(model_name, pretrained, strategy, vocab_size)

        # re-enable Hugging Face warnings
        transformers.logging.set_verbosity_warning()

        return model, model.config.hidden_size
