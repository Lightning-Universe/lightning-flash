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
from types import FunctionType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import Metric

from flash.core.classification import ClassificationAdapterTask, Labels
from flash.core.data.process import Serializer
from flash.core.registry import FlashRegistry
from flash.text.classification.adapters import TRAINING_STRATEGIES
from flash.text.classification.backbones import TEXT_CLASSIFIER_BACKBONES
from flash.text.ort_callback import ORTCallback


class TextClassifier(ClassificationAdapterTask):

    backbones: FlashRegistry = TEXT_CLASSIFIER_BACKBONES
    training_strategies: FlashRegistry = TRAINING_STRATEGIES

    required_extras: str = "text"

    def __init__(
        self,
        num_classes: Optional[int] = None,
        vocab_size: Optional[int] = None,
        backbone: Union[str, Tuple[nn.Module, int]] = "prajjwal1/bert-medium",
        backbone_kwargs: Optional[Dict] = None,
        head: Optional[Union[FunctionType, nn.Module]] = None,
        pretrained: Union[bool, str] = True,
        loss_fn: Optional[Callable] = None,
        optimizer: Union[Type[torch.optim.Optimizer], torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        metrics: Union[Metric, Callable, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-3,
        multi_label: bool = False,
        enable_ort: bool = False,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
        training_strategy: Optional[str] = "default",
        training_strategy_kwargs: Optional[Dict[str, Any]] = None,
    ):

        self.save_hyperparameters()

        self.enable_ort = enable_ort

        if not backbone_kwargs:
            backbone_kwargs = {}

        if not training_strategy_kwargs:
            training_strategy_kwargs = {}

        if training_strategy == "default":
            if not num_classes:
                raise MisconfigurationException("`num_classes` should be provided.")
            
        else:
            num_classes = training_strategy_kwargs.get("ways", None)
            if not num_classes:
                raise MisconfigurationException("`training_strategy_kwargs` should contain `ways`.")

        if isinstance(backbone, tuple):
            backbone, num_features = backbone
        else:
            backbone, num_features = self.backbones.get(backbone)(pretrained=pretrained, **backbone_kwargs)

        if backbone.vocab_size != vocab_size:
            raise MisconfigurationException("Model and tokenizer have different `vocab_size`.")

        head = head(num_features, num_classes) if isinstance(head, FunctionType) else head
        head = head or nn.Sequential(
            nn.Linear(num_features, num_classes),
        )

        adapter_from_class = self.training_strategies.get(training_strategy)
        adapter = adapter_from_class(
            task=self,
            num_classes=num_classes,
            backbone=backbone,
            head=head,
            pretrained=pretrained,
            **training_strategy_kwargs,
        )

        super().__init__(
            adapter,
            num_classes=num_classes,
            loss_fn=loss_fn,
            metrics=metrics,
            learning_rate=learning_rate,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            multi_label=multi_label,
            serializer=serializer or Labels(multi_label=multi_label),
        )

    def configure_callbacks(self) -> List[Callback]:
        callbacks = super().configure_callbacks() or []
        if self.enable_ort:
            callbacks.append(ORTCallback())
        return callbacks
