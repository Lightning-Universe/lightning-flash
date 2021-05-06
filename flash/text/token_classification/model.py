import os
import warnings
from typing import Callable, Mapping, Optional, Sequence, Type, Union

import torch
from transformers import AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput

from flash.core.classification import ClassificationTask
from flash.data.process import Serializer
from flash.text.token_classification.data import LABEL_IGNORE


class TokenClassifier(ClassificationTask):
    """Task that classifies tokens.

    Example::

        from flash.text import TokenClassifier

        task = TokenClassifier(num_classes=42, backbone="bert-base-uncased")

    Args:
        num_classes: Number of classes to classify.
        backbone: A token classification model from huggingface/transformers.
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `1e-3`
        multi_label: Whether the targets are multi-label or not.
        serializer: The :class:`~flash.data.process.Serializer` to use when serializing prediction outputs.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "prajjwal1/bert-tiny",
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[Callable, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-3,
        multi_label: bool = False,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
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
            multi_label=multi_label,
            serializer=serializer,
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            backbone, num_labels=num_classes
        )

    @property
    def backbone(self):
        return self.model.bert if hasattr(self.model, "bert") else self.model

    def forward(self, batch_dict):
        return self.model(**batch_dict)

    def step(self, batch, batch_idx) -> dict:
        out = self.forward(batch)
        loss, logits = out[:2]
        if isinstance(logits, TokenClassifierOutput):
            logits = logits.logits

        preds = logits.argmax(-1)

        labels = batch["labels"]
        mask = labels != LABEL_IGNORE

        preds_metric = preds[mask]
        labels_metric = labels[mask]
        output = {
            "loss": loss,
            "y_hat": logits,
            "logs": {
                name: metric(preds_metric, labels_metric)
                for name, metric in self.metrics.items()
            },
        }

        return output
