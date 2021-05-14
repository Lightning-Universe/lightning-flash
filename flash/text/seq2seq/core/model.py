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
from typing import Any, Callable, List, Mapping, Optional, Sequence, Type, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_info
from torch import Tensor

from flash.core.finetuning import FlashBaseFinetuning
from flash.core.model import Task
from flash.core.utilities.imports import _TEXT_AVAILABLE
from flash.text.seq2seq.core.finetuning import Seq2SeqFreezeEmbeddings

if _TEXT_AVAILABLE:
    from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerBase
else:
    AutoModelForSeq2SeqLM, PreTrainedTokenizerBase = None, None


def _pad_tensors_to_max_len(model_cfg, tensor, max_length):
    pad_token_id = model_cfg.pad_token_id if model_cfg.pad_token_id else model_cfg.eos_token_id
    if pad_token_id is None:
        raise ValueError(
            f"Make sure that either `config.pad_token_id` or `config.eos_token_id` "
            f"is defined if tensor has to be padded to `max_length`={max_length}"
        )

    padded_tensor = pad_token_id * torch.ones((tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device)
    padded_tensor[:, :tensor.shape[-1]] = tensor
    return padded_tensor


class Seq2SeqTask(Task):
    """General Task for Sequence2Sequence.
    Args:
        loss_fn: Loss function for training
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `3e-4`
        val_target_max_length: Maximum length of targets in validation. Defaults to `128`
        num_beams: Number of beams to use in validation when generating predictions. Defaults to `4`
    """

    def __init__(
        self,
        backbone: str = 't5-small',
        loss_fn: Optional[Union[Callable, Mapping, Sequence]] = None,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        metrics: Union[pl.metrics.Metric, Mapping, Sequence, None] = None,
        learning_rate: float = 5e-5,
        val_target_max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ):
        if not _TEXT_AVAILABLE:
            raise ModuleNotFoundError("Please, pip install -e '.[text]'")

        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
        # disable HF thousand warnings
        warnings.simplefilter("ignore")
        # set os environ variable for multiprocesses
        os.environ["PYTHONWARNINGS"] = "ignore"
        super().__init__(loss_fn=loss_fn, optimizer=optimizer, metrics=metrics, learning_rate=learning_rate)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(backbone)
        self.val_target_max_length = val_target_max_length
        self.num_beams = num_beams
        self._initialize_model_specific_parameters()

    def forward(self, x: Any) -> Any:
        max_length = self.val_target_max_length if self.val_target_max_length else self.model.config.max_length
        num_beams = self.num_beams if self.num_beams else self.model.config.num_beams
        generated_tokens = self.model.generate(
            input_ids=x['input_ids'], attention_mask=x['attention_mask'], max_length=max_length, num_beams=num_beams
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < max_length:
            generated_tokens = _pad_tensors_to_max_len(
                model_cfg=self.model.config, tensor=generated_tokens, max_length=max_length
            )
        return generated_tokens

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        generated_tokens = self(batch)
        self.compute_metrics(generated_tokens, batch, prefix)

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        self.common_step("val", batch)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        self.common_step("test", batch)

    def compute_metrics(self, generated_tokens, batch, prefix):
        pass

    @property
    def task(self) -> Optional[str]:
        """
        Override to define AutoConfig task specific parameters stored within the model.
        """
        return

    def _initialize_model_specific_parameters(self):
        task_specific_params = self.model.config.task_specific_params

        if task_specific_params:
            pars = task_specific_params.get(self.task, {})
            rank_zero_info(f"Overriding model paramameters for {self.task} as defined within the model:\n {pars}")
            self.model.config.update(pars)

    @property
    def tokenizer(self) -> 'PreTrainedTokenizerBase':
        return self.data_pipeline.data_source.tokenizer

    def tokenize_labels(self, labels: Tensor) -> List[str]:
        label_str = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        return [str.strip(s) for s in label_str]

    def configure_finetune_callback(self) -> List[FlashBaseFinetuning]:
        return [Seq2SeqFreezeEmbeddings(self.model.config.model_type, train_bn=True)]
