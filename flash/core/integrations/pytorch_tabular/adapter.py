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
from typing import Any, Callable, Dict, List, Optional, Union

import torchmetrics

from flash.core.adapter import Adapter
from flash.core.data.io.input import DataKeys
from flash.core.model import Task


class PytorchTabularAdapter(Adapter):
    def __init__(self, task_type, backbone):
        super().__init__()

        self.task_type = task_type
        self.backbone = backbone

    @classmethod
    def from_task(
        cls,
        task: Task,
        task_type,
        embedding_sizes: list,
        categorical_fields: list,
        cat_dims: list,
        num_features: int,
        output_dim: int,
        backbone: str,
        loss_fn: Optional[Callable],
        metrics: Optional[Union[torchmetrics.Metric, List[torchmetrics.Metric]]],
        backbone_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Adapter:

        backbone_kwargs = backbone_kwargs or {}
        parameters = {
            "embedding_dims": embedding_sizes,
            "categorical_cols": categorical_fields,
            "categorical_cardinality": cat_dims,
            "categorical_dim": len(categorical_fields),
            "continuous_dim": num_features - len(categorical_fields),
            "output_dim": output_dim,
        }
        adapter = cls(
            task_type,
            task.backbones.get(backbone)(
                task_type=task_type, parameters=parameters, loss_fn=loss_fn, metrics=metrics, **backbone_kwargs
            ),
        )

        return adapter

    def convert_batch(self, batch):
        new_batch = {
            "continuous": batch[DataKeys.INPUT][1],
            "categorical": batch[DataKeys.INPUT][0],
        }
        if DataKeys.TARGET in batch:
            new_batch["target"] = batch[DataKeys.TARGET].reshape(-1, 1)
            if self.task_type == "regression":
                new_batch["target"] = new_batch["target"].float()
        return new_batch

    def training_step(self, batch, batch_idx) -> Any:
        return self.backbone.training_step(self.convert_batch(batch), batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.backbone.validation_step(self.convert_batch(batch), batch_idx)

    def test_step(self, batch, batch_idx):
        return self.backbone.validation_step(self.convert_batch(batch), batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self(self.convert_batch(batch))

    def forward(self, batch: Any) -> Any:
        return self.backbone(batch)["logits"]
