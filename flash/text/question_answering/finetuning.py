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
import pytorch_lightning as pl

from flash.core.finetuning import FlashBaseFinetuning


class QuestionAnsweringFreezeEmbeddings(FlashBaseFinetuning):
    """Freezes the embedding layers during Question Answering training."""

    def __init__(self, model_type: str, train_bn: bool = True):
        super().__init__("", train_bn)
        self.model_type = model_type

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        if self.model_type in ["albert", "reformer"]:
            if self.model_type == "albert":
                model = pl_module.model.albert
            elif self.model_type == "reformer":
                model = pl_module.model.reformer

            self.freeze(model.embeddings, train_bn=self.train_bn)
            self.freeze(model.encoder, train_bn=self.train_bn)

        elif self.model_type in ["bart", "bigbird_pegasus", "led", "mbart"]:
            if self.model_type in ["bart", "bigbird_pegasus", "mbart"]:
                model = pl_module.model.model
            elif self.model_type == "led":
                model = pl_module.model.led

            self.freeze(modules=model.shared, train_bn=self.train_bn)
            self.freeze(model.encoder, train_bn=self.train_bn)
            self.freeze(model.decoder, train_bn=self.train_bn)

        elif self.model_type == [
            "bert",
            "big_bird",
            "ibert",
            "longformer",
            "lxmert",
            "mpnet",
            "megatron-bert",
            "mobilebert",
            "camembert",
            "roberta",
            "squeezebert",
            "xlm-roberta",
        ]:
            if self.model_type in ["bert", "big_bird", "megatron-bert"]:
                model = pl_module.model.bert
            elif self.model_type in ["camembert", "roberta", "xlm-roberta"]:
                model = pl_module.model.roberta
            elif self.model_type == "ibert":
                model = pl_module.model.ibert
            elif self.model_type == "longformer":
                model = pl_module.model.longformer
            elif self.model_type == "lxmert":
                model = pl_module.model.lxmert
            elif self.model_type == "mpnet":
                model = pl_module.model.mpnet
            elif self.model_type == "mobilebert":
                model = pl_module.model.mobilebert
            elif self.model_type == "squeezebert":
                model = pl_module.model.transformer

            self.freeze(model.embeddings, train_bn=self.train_bn)
            self.freeze(model.encoder, train_bn=self.train_bn)
            if model.pooler is not None:
                self.freeze(model.pooler, train_bn=self.train_bn)

        elif self.model_type == "canine":
            model = pl_module.model.canine

            self.freeze(model.char_embeddings, train_bn=self.train_bn)
            self.freeze(model.initial_char_encoder, train_bn=self.train_bn)
            self.freeze(model.chars_to_molecules, train_bn=self.train_bn)
            self.freeze(model.encoder, train_bn=self.train_bn)
            self.freeze(model.projection, train_bn=self.train_bn)
            self.freeze(model.final_char_encoder, train_bn=self.train_bn)
            if model.pooler is not None:
                self.freeze(model.pooler, train_bn=self.train_bn)

        elif self.model_type in ["convbert", "electra", "roformer"]:
            if self.model_type == "convbert":
                model = pl_module.model.convbert
            elif self.model_type == "electra":
                model = pl_module.model.electra
            elif self.model_type == "roformer":
                model = pl_module.model.roformer

            self.freeze(model.embeddings, train_bn=self.train_bn)
            if model.config.embedding_size != model.config.hidden_size:
                self.freeze(model.embeddings_project, train_bn=self.train_bn)
            self.freeze(model.encoder, train_bn=self.train_bn)

        elif self.model_type in ["deberta", "deberta-v2"]:
            model = pl_module.model.deberta

            self.freeze(model.embeddings, train_bn=self.train_bn)
            self.freeze(model.encoder, train_bn=self.train_bn)

        elif self.model_type == "distilbert":
            model = pl_module.model.distilbert

            self.freeze(model.embeddings, train_bn=self.train_bn)
            self.freeze(model.transformer, train_bn=self.train_bn)

        elif self.model_type in ["flaubert", "xlm"]:
            model = pl_module.model.transformer

            self.freeze(model.attentions, train_bn=self.train_bn)
            self.freeze(model.layer_norm1, train_bn=self.train_bn)
            self.freeze(model.ffns, train_bn=self.train_bn)
            self.freeze(model.layer_norm2, train_bn=self.train_bn)

        elif self.model_type == "funnel":
            model = pl_module.model.funnel

            self.freeze(model.embeddings, train_bn=self.train_bn)
            self.freeze(model.encoder, train_bn=self.train_bn)
            self.freeze(model.decoder, train_bn=self.train_bn)

        elif self.model_type == "xlnet":
            model = pl_module.model.transformer

            self.freeze(model.word_embedding, train_bn=self.train_bn)
            self.freeze(model.mask_emb, train_bn=self.train_bn)
            for layer in model.layer:
                self.freeze(layer, train_bn=self.train_bn)
            self.freeze(model.dropout, train_bn=self.train_bn)

        return None
