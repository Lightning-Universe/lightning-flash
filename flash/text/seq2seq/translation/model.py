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

import pytorch_lightning as pl
import torch

from flash.text.seq2seq.core.model import Seq2SeqTask
from flash.text.seq2seq.translation.metric import BLEUScore


class TranslationTask(Seq2SeqTask):
    """Task for Sequence2Sequence Translation.

    Args:
        backbone: backbone model to use for the task.
        loss_fn: Loss function for training.
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `3e-4`
        val_target_max_length: Maximum length of targets in validation. Defaults to `128`
        num_beams: Number of beams to use in validation when generating predictions. Defaults to `4`
        n_gram: Maximum n_grams to use in metric calculation. Defaults to `4`
        smooth: Apply smoothing in BLEU calculation. Defaults to `True`
    """

    def __init__(
        self,
        backbone: str = "t5-small",
        loss_fn: Optional[Union[Callable, Mapping, Sequence]] = None,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[pl.metrics.Metric, Mapping, Sequence, None] = None,
        learning_rate: float = 3e-4,
        val_target_max_length: Optional[int] = 128,
        num_beams: Optional[int] = 4,
        n_gram: bool = 4,
        smooth: bool = False,
    ):
        self.save_hyperparameters()
        super().__init__(
            backbone=backbone,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
            val_target_max_length=val_target_max_length,
            num_beams=num_beams,
        )
        self.bleu = BLEUScore(
            n_gram=n_gram,
            smooth=smooth,
        )

    @property
    def task(self) -> str:
        return "translation"

    def compute_metrics(self, generated_tokens, batch, prefix):
        tgt_lns = self.tokenize_labels(batch["labels"])
        # wrap targets in list as score expects a list of potential references
        tgt_lns = [[reference] for reference in tgt_lns]
        result = self.bleu(generated_tokens, tgt_lns)
        self.log(f"{prefix}_bleu_score", result, on_step=False, on_epoch=True, prog_bar=True)

    def _ci_benchmark_fn(self, history: List[Dict[str, Any]]):
        """
        This function is used only for debugging usage with CI
        """
        # assert history[-1]["val_bleu_score"]
