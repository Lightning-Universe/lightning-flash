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
from typing import Callable, Mapping, Optional, Sequence, Type, Union

import pytorch_lightning as pl
import torch

from flash.text.seq2seq.core.model import Seq2SeqTask
from flash.text.seq2seq.summarization.metric import RougeMetric


class SummarizationTask(Seq2SeqTask):
    """Task for Seq2Seq Summarization.

    Args:
        backbone: backbone model to use for the task.
        loss_fn: Loss function for training.
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `3e-4`
        val_target_max_length: Maximum length of targets in validation. Defaults to `128`
        num_beams: Number of beams to use in validation when generating predictions. Defaults to `4`
        use_stemmer: Whether Porter stemmer should be used to strip word suffixes to improve matching.
        rouge_newline_sep: Add a new line at the beginning of each sentence in Rouge Metric calculation.
    """

    def __init__(
        self,
        backbone: str = "t5-small",
        loss_fn: Optional[Union[Callable, Mapping, Sequence]] = None,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[pl.metrics.Metric, Mapping, Sequence, None] = None,
        learning_rate: float = 5e-5,
        val_target_max_length: Optional[int] = None,
        num_beams: Optional[int] = 4,
        use_stemmer: bool = True,
        rouge_newline_sep: bool = True
    ):
        self.save_hyperparameters()
        super().__init__(
            backbone=backbone,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
            val_target_max_length=val_target_max_length,
            num_beams=num_beams
        )
        self.rouge = RougeMetric(
            rouge_newline_sep=rouge_newline_sep,
            use_stemmer=use_stemmer,
        )

    @property
    def task(self) -> str:
        return "summarization"

    def compute_metrics(self, generated_tokens, batch, prefix):
        tgt_lns = self.tokenize_labels(batch["labels"])
        result = self.rouge(generated_tokens, tgt_lns)
        self.log_dict(result, on_step=False, on_epoch=True)
