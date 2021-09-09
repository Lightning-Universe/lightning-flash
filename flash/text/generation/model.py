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
from typing import Optional, Union, Callable, Mapping, Type, Sequence, List, Dict, Any

from pytorch_lightning import Callback
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import Metric
import torch
from flash.core.utilities.imports import _TEXT_AVAILABLE

from flash.core.model import Task
from flash.text.ort_callback import ORTCallback

if _TEXT_AVAILABLE:
    from transformers import AutoModelWithLMHead, PreTrainedTokenizerBase
    from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput

else:
    GPT2Model, PreTrainedTokenizerBase = None, None

class TextGeneration(Task):
    """

    """
    required_extras: str = "text"

    def __init__(
            self,
            backbone: str = "gpt2",
            loss_fn: Optional[Union[Callable, Mapping, Sequence]] = None,
            optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
            metrics: Union[Metric, Callable, Mapping, Sequence, None] = None,
            learning_rate: float = 5e-5,
            val_target_max_length: Optional[int] = None,
            num_beams: Optional[int] = None,
            enable_ort: bool = False,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
            scheduler_kwargs: Optional[Dict[str, Any]] = None,
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
        # disable HF thousand warnings
        warnings.simplefilter("ignore")
        # set os environ variable for multiprocesses
        os.environ["PYTHONWARNINGS"] = "ignore"
        super().__init__(
            loss_fn=loss_fn,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            metrics=metrics,
            learning_rate=learning_rate
        )
        self.model = AutoModelWithLMHead.from_pretrained(backbone)
        self.enable_ort = enable_ort
        self.val_target_max_length = val_target_max_length
        self.num_beams = num_beams
        self._initialize_model_specific_parameters()

    @property
    def backbone(self):
        return self.model.base_model

    def forward(self, batch: Dict[str, torch.Tensor]):
        return self.model(input_ids=batch.get("input_ids", None), attention_mask=batch.get("attention_mask", None))

    def to_loss_format(self, x) -> torch.Tensor:
        if isinstance(x, GPT2DoubleHeadsModelOutput):
            x = x.logits
        return super().to_loss_format(x)

    def to_metrics_format(self, x) -> torch.Tensor:
        if isinstance(x, GPT2DoubleHeadsModelOutput):
            x = x.logits
        return super().to_metrics_format(x)

    def step(self, batch, batch_idx, metrics) -> dict:
        target = batch.pop("labels")
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
