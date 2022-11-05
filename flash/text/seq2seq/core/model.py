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
from typing import Any, Dict, Iterable, List, Optional, Type, Union

import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_info
from torch import Tensor
from torch.nn import Module

from flash.core.data.io.input import DataKeys, ServeInput
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.io.output import Output
from flash.core.data.io.output_transform import OutputTransform
from flash.core.model import Task
from flash.core.registry import ExternalRegistry, FlashRegistry
from flash.core.serve import Composition
from flash.core.utilities.imports import _TEXT_AVAILABLE, requires
from flash.core.utilities.providers import _HUGGINGFACE
from flash.core.utilities.types import (
    INPUT_TRANSFORM_TYPE,
    LOSS_FN_TYPE,
    LR_SCHEDULER_TYPE,
    METRICS_TYPE,
    OPTIMIZER_TYPE,
)
from flash.text.input import TextDeserializer
from flash.text.ort_callback import ORTCallback
from flash.text.seq2seq.core.collate import TextSeq2SeqCollate

if _TEXT_AVAILABLE:
    from transformers import AutoModelForSeq2SeqLM

    HUGGINGFACE_BACKBONES = ExternalRegistry(
        AutoModelForSeq2SeqLM.from_pretrained,
        "backbones",
        _HUGGINGFACE,
    )
else:
    AutoModelForSeq2SeqLM, PreTrainedTokenizerBase = None, None

    HUGGINGFACE_BACKBONES = FlashRegistry("backbones")


def _pad_tensors_to_max_len(model_cfg, tensor, max_length):
    pad_token_id = model_cfg.pad_token_id if model_cfg.pad_token_id else model_cfg.eos_token_id
    if pad_token_id is None:
        raise ValueError(
            f"Make sure that either `config.pad_token_id` or `config.eos_token_id` "
            f"is defined if tensor has to be padded to `max_length`={max_length}"
        )

    padded_tensor = pad_token_id * torch.ones((tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device)
    padded_tensor[:, : tensor.shape[-1]] = tensor
    return padded_tensor


class Seq2SeqTask(Task):
    """General Task for Sequence2Sequence.

    Args:
        max_source_length: The maximum length to pad / truncate input sequences to.
        max_target_length: The maximum length to pad / truncate target sequences to.
        padding: The type of padding to apply. One of: "longest" or ``True``, "max_length", "do_not_pad" or
            ``False``.
        loss_fn: Loss function for training
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        metrics: Metrics to compute for training and evaluation. Changing this argument currently has no effect
        learning_rate: Learning rate to use for training, defaults to `3e-4`
        num_beams: Number of beams to use in validation when generating predictions. Defaults to `4`
        enable_ort: Enable Torch ONNX Runtime Optimization: https://onnxruntime.ai/docs/#onnx-runtime-for-training
    """

    required_extras: str = "text"

    backbones: FlashRegistry = FlashRegistry("backbones") + HUGGINGFACE_BACKBONES

    def __init__(
        self,
        backbone: str = "t5-small",
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = "max_length",
        loss_fn: LOSS_FN_TYPE = None,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        metrics: METRICS_TYPE = None,
        learning_rate: Optional[float] = None,
        num_beams: Optional[int] = None,
        enable_ort: bool = False,
        output_transform: Optional[OutputTransform] = None,
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
        # disable HF thousand warnings
        warnings.simplefilter("ignore")
        # set os environ variable for multiprocesses
        os.environ["PYTHONWARNINGS"] = "ignore"
        super().__init__(
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            learning_rate=learning_rate,
            output_transform=output_transform,
        )

        self.collate_fn = TextSeq2SeqCollate(
            backbone=backbone,
            tokenizer_kwargs=tokenizer_kwargs,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
        )
        self.model = self.backbones.get(backbone)()
        self.enable_ort = enable_ort
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.num_beams = num_beams
        self._initialize_model_specific_parameters()

    def forward(self, x: Any) -> Any:
        max_length = self.max_target_length
        num_beams = self.num_beams if self.num_beams else self.model.config.num_beams
        generated_tokens = self.model.generate(
            input_ids=x["input_ids"],
            attention_mask=x.get("attention_mask", None),
            max_length=max_length,
            num_beams=num_beams,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < max_length:
            generated_tokens = _pad_tensors_to_max_len(
                model_cfg=self.model.config, tensor=generated_tokens, max_length=max_length
            )
        return generated_tokens

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        batch["labels"] = batch.pop(DataKeys.TARGET)
        outputs = self.model(**batch)
        loss = outputs[0]
        self.log("train_loss", loss)
        return loss

    def common_step(self, prefix: str, batch: Any) -> Tensor:
        batch["labels"] = batch.pop(DataKeys.TARGET)
        generated_tokens = self(batch)
        self.compute_metrics(generated_tokens, batch, prefix)

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        self.common_step("val", batch)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        self.common_step("test", batch)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        output = super().predict_step(batch, batch_idx, dataloader_idx)
        return self.decode(output)

    def compute_metrics(self, generated_tokens, batch, prefix):
        pass

    @property
    def task(self) -> Optional[str]:
        """Override to define AutoConfig task specific parameters stored within the model."""
        return

    def _initialize_model_specific_parameters(self):
        task_specific_params = self.model.config.task_specific_params

        if task_specific_params:
            pars = task_specific_params.get(self.task, {})
            rank_zero_info(f"Overriding model paramameters for {self.task} as defined within the model:\n {pars}")
            self.model.config.update(pars)

    def decode(self, tokens: Tensor) -> List[str]:
        decoded_str = self.collate_fn.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        return [str.strip(s) for s in decoded_str]

    def modules_to_freeze(self) -> Union[Module, Iterable[Union[Module, Iterable]]]:
        """Return the module attributes of the model to be frozen."""
        model_type = self.model.config.model_type

        _modules = []

        is_t5 = model_type in ["t5", "mt5"]
        model = self.model if is_t5 else self.model.model
        _modules.append(model.shared)
        for layer in (model.encoder, model.decoder):
            _modules.append(layer.embed_tokens)
            if not is_t5:
                _modules.append(layer.embed_positions)
        return _modules

    def configure_callbacks(self) -> List[Callback]:
        callbacks = super().configure_callbacks() or []
        if self.enable_ort:
            callbacks.append(ORTCallback())
        return callbacks

    @requires("serve")
    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        sanity_check: bool = True,
        input_cls: Optional[Type[ServeInput]] = TextDeserializer,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        output: Optional[Union[str, Output]] = None,
    ) -> Composition:
        return super().serve(host, port, sanity_check, input_cls, transform, transform_kwargs, output)
