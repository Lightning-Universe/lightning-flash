from typing import Callable, Mapping, Sequence, Union, Type

import torch
from torch import nn
import torch.nn.functional as F

from pl_flash import Model

from transformers import BertForSequenceClassification


class TextClassifier(Model):
    """Task that classifies text.

    Args:
        num_classes: Number of classes to classify.
        backbone: A model to use to compute text features can be any BERT model from HuggingFace/transformersimage .
        pretrained: Use a pretrained backbone.
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `1e-3`
    """

    def __init__(
        self,
        num_classes,
        backbone="bert-base-uncased",
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[Callable, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-5,
    ):

        super().__init__(
            model=None,
            loss_fn=None,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
        )
        self.model = BertForSequenceClassification.from_pretrained(backbone, num_labels=num_classes)

    def forward(self, batch_dict):
        loss, logits = self.model(**batch_dict)[:2]
        return loss, logits

    def step(self, batch, batch_idx):
        loss, logits = self.forward(batch)
        labels = batch["labels"]
        logs = {name: metric(logits, labels) for name, metric in self.metrics.items()}
        logs["loss"] = loss
        return loss, logs
