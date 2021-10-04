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
from typing import Union

from pytorch_lightning.core.lightning import LightningModule

from flash.core.model import Task


def _get_question_answering_bacbones_for_freezing(pl_module: Union[Task, LightningModule]):
    model_type = pl_module.config.model_type

    if model_type in ["albert", "reformer"]:
        if model_type == "albert":
            model = pl_module.model.albert
        elif model_type == "reformer":
            model = pl_module.model.reformer

    elif model_type in ["bart", "bigbird_pegasus", "led", "mbart"]:
        if model_type in ["bart", "bigbird_pegasus", "mbart"]:
            model = pl_module.model.model
        elif model_type == "led":
            model = pl_module.model.led

    elif model_type == [
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
        if model_type in ["bert", "big_bird", "megatron-bert"]:
            model = pl_module.model.bert
        elif model_type in ["camembert", "roberta", "xlm-roberta"]:
            model = pl_module.model.roberta
        elif model_type == "ibert":
            model = pl_module.model.ibert
        elif model_type == "longformer":
            model = pl_module.model.longformer
        elif model_type == "lxmert":
            model = pl_module.model.lxmert
        elif model_type == "mpnet":
            model = pl_module.model.mpnet
        elif model_type == "mobilebert":
            model = pl_module.model.mobilebert
        elif model_type == "squeezebert":
            model = pl_module.model.transformer

    elif model_type == "canine":
        model = pl_module.model.canine

    elif model_type in ["convbert", "electra", "roformer"]:
        if model_type == "convbert":
            model = pl_module.model.convbert
        elif model_type == "electra":
            model = pl_module.model.electra
        elif model_type == "roformer":
            model = pl_module.model.roformer

    elif model_type in ["deberta", "deberta-v2"]:
        model = pl_module.model.deberta

    elif model_type == "distilbert":
        model = pl_module.model.distilbert

    elif model_type in ["flaubert", "xlm"]:
        model = pl_module.model.transformer

    elif model_type == "funnel":
        model = pl_module.model.funnel

    elif model_type == "xlnet":
        model = pl_module.model.transformer
    else:
        # TODO: Is this the right way to exit the if statement ?
        model = None

    return model
