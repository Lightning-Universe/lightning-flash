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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Type, Union

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import Metric

from flash.text.seq2seq.core.metrics import RougeMetric
from flash.text.seq2seq.core.model import Seq2SeqTask


class SummarizationTask(Seq2SeqTask):
    """The ``SummarizationTask`` is a :class:`~flash.Task` for Seq2Seq text summarization. For more details, see
    :ref:`summarization`.

    You can change the backbone to any summarization model from `HuggingFace/transformers
    <https://huggingface.co/models?filter=pytorch&pipeline_tag=summarization>`_ using the ``backbone`` argument.

    .. note:: When changing the backbone, make sure you pass in the same backbone to the :class:`~flash.Task` and the
        :class:`~flash.core.data.data_module.DataModule` object! Since this is a Seq2Seq task, make sure you use a
        Seq2Seq model.

    Args:
        backbone: backbone model to use for the task.
        loss_fn: Loss function for training.
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        optimizer_kwargs: Additional kwargs to use when creating the optimizer (if not passed as an instance).
        scheduler: The scheduler or scheduler class to use.
        scheduler_kwargs: Additional kwargs to use when creating the scheduler (if not passed as an instance).
        metrics: Metrics to compute for training and evaluation. Defauls to calculating the ROUGE metric.
            Changing this argument currently has no effect.
        learning_rate: Learning rate to use for training, defaults to `3e-4`
        val_target_max_length: Maximum length of targets in validation. Defaults to `128`
        num_beams: Number of beams to use in validation when generating predictions. Defaults to `4`
        use_stemmer: Whether Porter stemmer should be used to strip word suffixes to improve matching.
        rouge_newline_sep: Add a new line at the beginning of each sentence in Rouge Metric calculation.
        enable_ort: Enable Torch ONNX Runtime Optimization: https://onnxruntime.ai/docs/#onnx-runtime-for-training
    """

    def __init__(
        self,
        backbone: str = "sshleifer/distilbart-xsum-1-1",
        loss_fn: Optional[Union[Callable, Mapping, Sequence]] = None,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        metrics: Union[Metric, Callable, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-5,
        val_target_max_length: Optional[int] = None,
        num_beams: Optional[int] = 4,
        use_stemmer: bool = True,
        rouge_newline_sep: bool = True,
        enable_ort: bool = False,
    ):
        self.save_hyperparameters()
        super().__init__(
            backbone=backbone,
            loss_fn=loss_fn,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            metrics=metrics,
            learning_rate=learning_rate,
            val_target_max_length=val_target_max_length,
            num_beams=num_beams,
            enable_ort=enable_ort,
        )
        self.rouge = RougeMetric(
            rouge_newline_sep=rouge_newline_sep,
            use_stemmer=use_stemmer,
        )

    @property
    def task(self) -> str:
        return "summarization"

    def compute_metrics(self, generated_tokens: torch.Tensor, batch: Dict, prefix: str) -> None:
        tgt_lns = self.tokenize_labels(batch["labels"])
        result = self.rouge(self._postprocess.uncollate(generated_tokens), tgt_lns)
        self.log_dict(result, on_step=False, on_epoch=True, prog_bar=True)

    @staticmethod
    def _ci_benchmark_fn(history: List[Dict[str, Any]]):
        """This function is used only for debugging usage with CI."""
        assert history[-1]["rouge1_recall"] > 0.2
