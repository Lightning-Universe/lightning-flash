from typing import Callable, Mapping, Sequence, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertForSequenceClassification

from pl_flash import ClassificationLightningTask


class TextClassifier(ClassificationLightningTask):
    """LightningTask that classifies text.

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
        output = {}
        x, target = batch["x"], batch["target"]
        loss, logits = self.forward(x)
        output["loss"] = loss
        output["y_hat"] = logits
        output["logs"] = {name: metric(logits, target) for name, metric in self.metrics.items()}
        return output
