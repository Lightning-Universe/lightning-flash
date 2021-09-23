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
from typing import Any, Callable, List, Mapping, Optional, Sequence, Type, Union

import torch
from torch import nn
from torch.optim import Optimizer

from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.process import Preprocess, Serializer
from flash.core.model import Task
from flash.core.utilities.imports import _FASTFACE_AVAILABLE
from flash.image.detection.finetuning import ObjectDetectionFineTuning
from flash.image.detection.serialization import DetectionLabels
from flash.image.face_detection.data import FaceDetectionPreprocess

if _FASTFACE_AVAILABLE:
    import fastface as ff


class FaceDetector(Task):
    """The ``FaceDetector`` is a :class:`~flash.Task` for detecting faces in images. For more details, see
    :ref:`face_detection`.
    Args:
        model: a string of :attr`_models`. Defaults to 'lffd_slim'.
        pretrained: Whether the model from fastface should be loaded with it's pretrained weights.
        loss: the function(s) to update the model with. Has no effect for fastface models.
        metrics: The provided metrics. All metrics here will be logged to progress bar and the respective logger.
            Changing this argument currently has no effect.
        optimizer: The optimizer to use for training. Can either be the actual class or the class name.
        learning_rate: The learning rate to use for training
    """

    required_extras: str = "image"

    def __init__(
        self,
        model: str = "lffd_slim",
        pretrained: bool = True,
        loss=None,
        metrics: Union[Callable, nn.Module, Mapping, Sequence, None] = None,
        optimizer: Type[Optimizer] = torch.optim.AdamW,
        learning_rate: float = 1e-4,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
        preprocess: Optional[Preprocess] = None,
        **kwargs: Any,
    ):
        self.save_hyperparameters()

        if model in ff.list_pretrained_models():
            model = FaceDetector.get_model(model, pretrained, **kwargs)
        else:
            ValueError(f"{model} is not supported yet.")

        super().__init__(
            model=model,
            loss_fn=loss,
            metrics=metrics or {"AP": ff.metric.AveragePrecision()},
            learning_rate=learning_rate,
            optimizer=optimizer,
            serializer=serializer or DetectionLabels(),
            preprocess=preprocess or FaceDetectionPreprocess(),
        )

    @staticmethod
    def get_model(
        model_name,
        pretrained,
        **kwargs,
    ):

        if pretrained:
            pl_model = ff.FaceDetector.from_pretrained(model_name, **kwargs)
        else:
            arch, config = model_name.split("_")
            pl_model = ff.FaceDetector.build(arch, config, **kwargs)

        # get torch.nn.Module
        model = getattr(pl_model, "arch")

        # set preprocess params
        model.register_buffer("normalizer", getattr(pl_model, "normalizer"))
        model.register_buffer("mean", getattr(pl_model, "mean"))
        model.register_buffer("std", getattr(pl_model, "std"))

        # set postprocess function
        setattr(model, "_postprocess", getattr(pl_model, "_postprocess"))

        return model

    def forward(self, x: List[torch.Tensor]) -> Any:

        batch, scales, paddings = ff.utils.preprocess.prepare_batch(x, None, adaptive_batch=True)
        # batch: torch.Tensor(B,C,T,T)
        # scales: torch.Tensor(B,)
        # paddings: torch.Tensor(B,4) as pad (left, top, right, bottom)

        # apply preprocess
        batch = (((batch * 255) / self.model.normalizer) - self.model.mean) / self.model.std

        # get logits
        logits = self.model(batch)
        # logits, any

        preds = self.model.logits_to_preds(logits)
        # preds: torch.Tensor(B, N, 5)

        preds = self.model._postprocess(preds)
        # preds: torch.Tensor(N, 6) as x1,y1,x2,y2,score,batch_idx

        preds = [preds[preds[:, 5] == batch_idx, :5] for batch_idx in range(batch.size(0))]
        # preds: list of torch.Tensor(N, 5) as x1,y1,x2,y2,score

        preds = ff.utils.preprocess.adjust_results(preds, scales, paddings)
        # preds: list of torch.Tensor(N, 5) as x1,y1,x2,y2,score

        return preds

    def _prepare_batch(self, batch):
        images, targets = batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET]

        targets = [{"target_boxes": target["boxes"]} for target in targets]

        batch, scales, paddings = ff.utils.preprocess.prepare_batch(images, None, adaptive_batch=True)
        # batch: torch.Tensor(B,C,T,T)
        # scales: torch.Tensor(B,)
        # paddings: torch.Tensor(B,4) as pad (left, top, right, bottom)

        # apply preprocess
        batch = (((batch * 255) / self.model.normalizer) - self.model.mean) / self.model.std

        # adjust targets
        for i, (target, scale, padding) in enumerate(zip(targets, scales, paddings)):
            target["target_boxes"] *= scale
            target["target_boxes"][:, [0, 2]] += padding[0]
            target["target_boxes"][:, [1, 3]] += padding[1]
            targets[i]["target_boxes"] = target["target_boxes"]

        return batch, targets

    def _compute_metrics(self, logits, targets):
        preds = self.model.logits_to_preds(logits)
        # preds: torch.Tensor(B, N, 5)

        preds = self.model._postprocess(preds)
        # preds: torch.Tensor(N, 6) as x1,y1,x2,y2,score,batch_idx

        target_boxes = [target["target_boxes"] for target in targets]
        pred_boxes = [preds[preds[:, 5] == batch_idx, :5] for batch_idx in range(len(targets))]

        for metric in self.val_metrics.values():
            metric.update(pred_boxes, target_boxes)

    def training_step(self, batch, batch_idx) -> Any:
        """The training step. Overrides ``Task.training_step``
        """

        batch, targets = self._prepare_batch(batch)

        # get logits
        logits = self.model(batch)
        # logits, any

        # compute loss
        loss = self.model.compute_loss(logits, targets)
        # loss: dict of losses or loss

        self.log_dict({f"train_{k}": v for k, v in loss.items()}, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        for metric in self.val_metrics.values():
            metric.reset()

    def validation_step(self, batch, batch_idx):
        batch, targets = self._prepare_batch(batch)

        # get logits
        logits = self.model(batch)
        # logits, any

        # compute loss
        loss = self.model.compute_loss(logits, targets)
        # loss: dict of losses or loss

        self._compute_metrics(logits, targets)

        self.log_dict({f"val_{k}": v for k, v in loss.items()}, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs) -> None:
        metric_results = {name: metric.compute() for name, metric in self.val_metrics.items()}
        self.log_dict({f"val_{k}": v for k, v in metric_results.items()}, on_epoch=True)

    def on_test_epoch_start(self) -> None:
        for metric in self.val_metrics.values():
            metric.reset()

    def test_step(self, batch, batch_idx):
        batch, targets = self._prepare_batch(batch)

        # get logits
        logits = self.model(batch)
        # logits, any

        # compute loss
        loss = self.model.compute_loss(logits, targets)
        # loss: dict of losses or loss

        self._compute_metrics(logits, targets)

        self.log_dict({f"test_{k}": v for k, v in loss.items()}, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_epoch_end(self, outputs) -> None:
        metric_results = {name: metric.compute() for name, metric in self.val_metrics.items()}
        self.log_dict({f"test_{k}": v for k, v in metric_results.items()}, on_epoch=True)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        images = batch[DefaultDataKeys.INPUT]
        batch[DefaultDataKeys.PREDS] = self(images)
        return batch

    def configure_finetune_callback(self):
        return [ObjectDetectionFineTuning(train_bn=True)]
