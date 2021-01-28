from typing import Callable, Dict, List, Mapping, Sequence, Type, Union

import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification, default_data_collator

from flash.core.classification import ClassificationTask
from flash.text.classification.data import prepare_dataset, TextClassificationData, tokenize_text_lambda


class TextClassifier(ClassificationTask):
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
        num_classes: int,
        backbone: str = "bert-base-uncased",
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[Callable, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-5,
    ):
        self.save_hyperparameters()

        self._predict = False

        super().__init__(
            model=None,
            loss_fn=None,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
        )
        self.model = BertForSequenceClassification.from_pretrained(backbone, num_labels=num_classes)

    @property
    def backbone(self):
        # see huggingface's BertForSequenceClassification
        return self.model.bert

    def forward(self, batch_dict):
        return self.model(**batch_dict)

    def step(self, batch, batch_idx):
        output = {}
        loss, logits = self.forward(batch)
        if self._predict:
            return logits
        output["loss"] = loss
        output["y_hat"] = logits
        output["logs"] = {name: metric(logits, batch["labels"]) for name, metric in self.metrics.items()}
        return output

    @staticmethod
    def default_pipeline():
        return TextClassificationData.default_pipeline()
