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
import inspect
from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning import Callback

from flash.core.classification import ClassificationTask, Labels
from flash.core.data.io.input import DataKeys
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _TRANSFORMERS_AVAILABLE
from flash.core.utilities.types import LOSS_FN_TYPE, LR_SCHEDULER_TYPE, METRICS_TYPE, OPTIMIZER_TYPE, OUTPUT_TYPE
from flash.text.classification.backbones import TEXT_CLASSIFIER_BACKBONES
from flash.text.ort_callback import ORTCallback

if _TRANSFORMERS_AVAILABLE:
    from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput, SequenceClassifierOutput


class TextClassifier(ClassificationTask):
    """The ``TextClassifier`` is a :class:`~flash.Task` for classifying text. For more details, see
    :ref:`text_classification`. The ``TextClassifier`` also supports multi-label classification with
    ``multi_label=True``. For more details, see :ref:`text_classification_multi_label`.

    Args:
        num_classes: Number of classes to classify.
        backbone: A model to use to compute text features can be any BERT model from HuggingFace/transformersimage .
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        metrics: Metrics to compute for training and evaluation. Can either be an metric from the `torchmetrics`
            package, a custom metric inherenting from `torchmetrics.Metric`, a callable function or a list/dict
            containing a combination of the aforementioned. In all cases, each metric needs to have the signature
            `metric(preds,target)` and return a single scalar tensor. Defaults to :class:`torchmetrics.Accuracy`.
        learning_rate: Learning rate to use for training, defaults to `1e-3`
        multi_label: Whether the targets are multi-label or not.
        output: The :class:`~flash.core.data.io.output.Output` to use when formatting prediction outputs.
        enable_ort: Enable Torch ONNX Runtime Optimization: https://onnxruntime.ai/docs/#onnx-runtime-for-training
    """

    required_extras: str = "text"

    backbones: FlashRegistry = TEXT_CLASSIFIER_BACKBONES

    def __init__(
        self,
        num_classes: int,
        backbone: str = "prajjwal1/bert-medium",
        vocab_size: Optional[int] = None,
        pretrained: Optional[bool] = True,
        loss_fn: LOSS_FN_TYPE = None,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        metrics: METRICS_TYPE = None,
        learning_rate: float = 1e-2,
        multi_label: bool = False,
        output: OUTPUT_TYPE = None,
        enable_ort: bool = False,
    ):

        super().__init__(
            num_classes=num_classes,
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            learning_rate=learning_rate,
            multi_label=multi_label,
            output=output or Labels(multi_label=multi_label),
        )
        self.enable_ort = enable_ort
        self.model = self.backbones.get(backbone)(num_labels=num_classes)
        self.pretrained = pretrained

        if self.pretrained:
            self.vocab_size = self.model.config.vocab_size
            print("`pretrained=True`, ignoring `vocab_size` argument.")

        else:
            self.vocab_size = vocab_size if vocab_size else self.model.config.vocab_size
            replace_embeddings(self.model, self.vocab_size, self.model.config.vocab_size)
            print(f"Re-initialized word embeddings layer with `vocab_size={self.vocab_size}`")

        self.save_hyperparameters()

    @property
    def backbone(self):
        return self.model.base_model

    def forward(self, batch: Dict[str, torch.Tensor]):
        result = self.model(input_ids=batch.get("input_ids", None), attention_mask=batch.get("attention_mask", None))
        if isinstance(result, (SequenceClassifierOutput, Seq2SeqSequenceClassifierOutput)):
            result = result.logits
        return result

    def step(self, batch, batch_idx, metrics) -> dict:
        target = batch.pop(DataKeys.TARGET)
        batch = (batch, target)
        return super().step(batch, batch_idx, metrics)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self(batch)

    def _ci_benchmark_fn(self, history: List[Dict[str, Any]]):
        """This function is used only for debugging usage with CI."""
        if self.hparams.multi_label:
            assert history[-1]["val_f1"] > 0.40, history[-1]["val_f1"]
        else:
            assert history[-1]["val_accuracy"] > 0.70, history[-1]["val_accuracy"]

    def configure_callbacks(self) -> List[Callback]:
        callbacks = super().configure_callbacks() or []
        if self.enable_ort:
            callbacks.append(ORTCallback())
        return callbacks


def replace_embeddings(module, vocab_size, original_vocab_size):
    """
    Recursively overwrites torch.nn.Embedding and torch.nn.EmbeddingBag layers in module
    by only changing the `num_embeddings` attribute and keeping everything else equal.

    It reinitializes the embedding modules using the Pytorch default initialization since
    it cannot be known what initialization was used for each backbone automatically.
    """

    # get object class input signature to reinitialize the object with only the vocab_size changed
    def get_inputs(target_cls):
        return {
            attr: attr_value
            for attr, attr_value in [
                (attr, getattr(target_attr, attr, None)) for attr in inspect.signature(target_cls).parameters.keys()
            ]
            if attr
        }

    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)

        for target_cls in [torch.nn.Embedding, torch.nn.EmbeddingBag]:

            # search module based on class and original vocab_size
            if type(target_attr) == target_cls and target_attr.weight.shape[0] == original_vocab_size:
                other_inputs = get_inputs(target_cls)
                other_inputs.pop("num_embeddings")
                new_bn = target_cls(num_embeddings=vocab_size, **other_inputs)

                setattr(module, attr_str, new_bn)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for _, immediate_child_module in module.named_children():
        replace_embeddings(immediate_child_module, vocab_size, original_vocab_size)
