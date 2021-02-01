from typing import Callable, Mapping, Optional, Sequence, Type, Union

import pytorch_lightning as pl
import torch

from flash.text.seq2seq.core.model import Seq2SeqTask
from flash.text.seq2seq.summarization.metric import RougeMetric


class SummarizationTask(Seq2SeqTask):

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
