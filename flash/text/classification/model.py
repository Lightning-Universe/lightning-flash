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
import os
import warnings
from typing import Callable, Mapping, Optional, Sequence, Type, Union

import torch
from torchmetrics import Accuracy
from transformers import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from flash.core.classification import ClassificationTask
# from flash.return_types.classification import ClassificationReturnType


class TextClassifier(ClassificationTask):
    """Task that classifies text.

    Args:
        num_classes: Number of classes to classify.
        backbone: A model to use to compute text features can be any BERT model from HuggingFace/transformersimage .
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `1e-3`
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "prajjwal1/bert-tiny",
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[Callable, Mapping, Sequence, None] = [Accuracy()],
        learning_rate: float = 1e-3,
        # serializer: Optional[Serializ] = None,
    ):
        self.save_hyperparameters()

        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
        # disable HF thousand warnings
        warnings.simplefilter("ignore")
        # set os environ variable for multiprocesses
        os.environ["PYTHONWARNINGS"] = "ignore"

        super().__init__(
            model=None,
            loss_fn=None,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
            # return_type=return_type,
        )
        self.model = BertForSequenceClassification.from_pretrained(backbone, num_labels=num_classes)

    @property
    def backbone(self):
        # see huggingface's BertForSequenceClassification
        return self.model.bert

    def forward(self, batch_dict):
        return self.model(**batch_dict)

    def step(self, batch, batch_idx) -> dict:
        output = {}
        out = self.forward(batch)
        loss, logits = out[:2]
        output["loss"] = loss
        output["y_hat"] = logits
        if isinstance(logits, SequenceClassifierOutput):
            logits = logits.logits
        probs = torch.softmax(logits, 1)
        output["logs"] = {name: metric(probs, batch["labels"]) for name, metric in self.metrics.items()}
        return output
