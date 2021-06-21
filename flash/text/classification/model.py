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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Type, Union

import torch

from flash.core.classification import ClassificationTask, Labels
from flash.core.data.process import Serializer
from flash.core.utilities.imports import _TEXT_AVAILABLE

if _TEXT_AVAILABLE:
    from transformers import BertForSequenceClassification
    from transformers.modeling_outputs import SequenceClassifierOutput


class TextClassifier(ClassificationTask):
    """Task that classifies text.

    Args:
        num_classes: Number of classes to classify.
        backbone: A model to use to compute text features can be any BERT model from HuggingFace/transformersimage .
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `1e-3`
        multi_label: Whether the targets are multi-label or not.
        serializer: The :class:`~flash.core.data.process.Serializer` to use when serializing prediction outputs.
    """

    required_extras: str = "text"

    def __init__(
        self,
        num_classes: int,
        backbone: str = "prajjwal1/bert-medium",
        loss_fn: Optional[Callable] = None,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[Callable, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-2,
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
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
            multi_label=multi_label,
            serializer=serializer or Labels(multi_label=multi_label),
        )
        self.model = BertForSequenceClassification.from_pretrained(backbone, num_labels=num_classes)

        self.save_hyperparameters()

    @property
    def backbone(self):
        # see huggingface's BertForSequenceClassification
        return self.model.bert

    def forward(self, batch: Dict[str, torch.Tensor]):
        return self.model(input_ids=batch.get("input_ids", None), attention_mask=batch.get("attention_mask", None))

    def to_loss_format(self, x) -> torch.Tensor:
        if isinstance(x, SequenceClassifierOutput):
            x = x.logits
        return super().to_loss_format(x)

    def to_metrics_format(self, x) -> torch.Tensor:
        if isinstance(x, SequenceClassifierOutput):
            x = x.logits
        return super().to_metrics_format(x)

    def step(self, batch, batch_idx) -> dict:
        target = batch.pop("labels")
        batch = (batch, target)
        return super().step(batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self(batch)

    def _ci_benchmark_fn(self, history: List[Dict[str, Any]]):
        """
        This function is used only for debugging usage with CI
        """
        if self.hparams.multi_label:
            assert history[-1]["val_f1"] > 0.45
        else:
            assert history[-1]["val_accuracy"] > 0.73
