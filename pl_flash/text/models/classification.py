from typing import Callable, Dict, List, Mapping, Sequence, Type, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, BertForSequenceClassification,
                          default_data_collator)

from pl_flash.model import ClassificationLightningTask
from pl_flash.text.data.classification import (prepare_dataset,
                                               tokenize_text_lambda)


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
        num_classes: int,
        backbone: str,
        label_to_class_mapping: Dict = None,
        max_length: int = None,
        text_field: str = None,
        label_field: str = None,
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

    def forward(self, batch_dict):
        if self._predict:
            return None, self.model(**batch_dict)[-1]
        else:
            return self.model(**batch_dict)[:2]

    def step(self, batch, batch_idx):
        output = {}
        loss, logits = self.forward(batch)
        if self._predict:
            return logits
        output["loss"] = loss
        output["y_hat"] = logits
        output["logs"] = {name: metric(logits, batch["labels"]) for name, metric in self.metrics.items()}
        return output

    def predict(self, sequences: Union[List[str], str] = None, path_to_csv: str = None,
                num_workers: int = 0, batch_size: int = 2, limit_test_batches: int = 8, **kwargs):

        self._predict = True

        if sequences and path_to_csv:
            raise MisconfigurationException(
                "sequences or path_to_csv are mutually exclusive. Provide one or the other.")

        if isinstance(path_to_csv, str):
            extension = path_to_csv.split('.')[-1]
            if extension != 'csv':
                raise MisconfigurationException(
                    "only `.csv` file are currently supported for inference.")

            _, _, test_ds, _ = prepare_dataset(
                None, None, path_to_csv, extension, self.hparams.backbone,
                self.hparams.text_field, self.hparams.max_length,
                label_field=self.hparams.label_field,
                label_to_class_mapping=self.hparams.label_to_class_mapping)

            collate_fn = None

        elif isinstance(sequences, list) and not sequences:
            tokenizer = AutoTokenizer.from_pretrained(self.hparams.backbone, use_fast=True)
            tokenize_fn = tokenize_text_lambda(tokenizer, self.hparams.text_field, self.hparams.max_length)
            test_ds = [tokenize_fn({self.hparams.text_field: s}) for s in sequences]
            collate_fn = default_data_collator
        else:
            raise MisconfigurationException(
                "sequences or path_to_csv should be provided to make an inference. Provide one or the other.")

        test_dataloaders = [DataLoader(test_ds, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)]

        trainer = pl.Trainer(limit_test_batches=limit_test_batches, **kwargs)

        results = trainer.test(self, test_dataloaders=test_dataloaders)

        return results
