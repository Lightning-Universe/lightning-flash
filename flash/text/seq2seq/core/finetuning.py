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
from flash.core.finetuning import FlashBaseFinetuning
from flash.text.seq2seq.core import Seq2SeqTask


class Seq2SeqFreezeEmbeddings(FlashBaseFinetuning):
    """
    Freezes the embedding layers during Seq2Seq training.
    """

    def __init__(self, train_bn: bool = True):
        super().__init__("", train_bn)

    def freeze_before_training(self, pl_module: Seq2SeqTask) -> None:
        model_type = pl_module.model.config.model_type
        is_t5 = model_type in ["t5", "mt5"]
        model = pl_module.model if is_t5 else pl_module.model.model
        self.freeze(module=model.shared, train_bn=self.train_bn)
        for layer in (model.encoder, model.decoder):
            self.freeze(layer.embed_tokens)
            if not is_t5:
                self.freeze(layer.embed_positions)
