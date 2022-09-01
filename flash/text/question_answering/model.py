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

# Adapted from:
# https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa_no_trainer.py
# https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/utils_qa.py

import collections
import os
import warnings
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_info
from torch import Tensor
from torch.nn import Module
from torchmetrics.text.rouge import ROUGEScore

from flash.core.data.io.input import DataKeys
from flash.core.model import Task
from flash.core.registry import ExternalRegistry, FlashRegistry
from flash.core.utilities.imports import _TEXT_AVAILABLE, _TM_GREATER_EQUAL_0_7_0
from flash.core.utilities.providers import _HUGGINGFACE
from flash.core.utilities.types import LR_SCHEDULER_TYPE, METRICS_TYPE, OPTIMIZER_TYPE
from flash.text.ort_callback import ORTCallback
from flash.text.question_answering.collate import TextQuestionAnsweringCollate
from flash.text.question_answering.output_transform import QuestionAnsweringOutputTransform

if _TEXT_AVAILABLE:
    from transformers import AutoModelForQuestionAnswering

    HUGGINGFACE_BACKBONES = ExternalRegistry(
        AutoModelForQuestionAnswering.from_pretrained,
        "backbones",
        _HUGGINGFACE,
    )
else:
    AutoModelForQuestionAnswering = None

    HUGGINGFACE_BACKBONES = FlashRegistry("backbones")


class QuestionAnsweringTask(Task):
    """The ``QuestionAnsweringTask`` is a :class:`~flash.Task` for extractive question answering. For more details,
    see `question_answering`.

    You can change the backbone to any question answering model from `HuggingFace/transformers
    <https://huggingface.co/transformers/model_doc/auto.html#automodelforquestionanswering>`_ using the ``backbone``
    argument.

    .. note:: When changing the backbone, make sure you pass in the same backbone to the :class:`~flash.Task` and the
        :class:`~flash.core.data.data_module.DataModule` object! Since this is a QuestionAnswering task, make sure you
        use a QuestionAnswering model.

    Args:
        backbone: backbone model to use for the task.
        max_source_length: Max length of the sequence to be considered during tokenization.
        max_target_length: Max length of each answer to be produced.
        padding: Padding type during tokenization.
        doc_stride: The stride amount to be taken when splitting up a long document into chunks.
        loss_fn: Loss function for training.
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        metrics: Metrics to compute for training and evaluation. Defauls to calculating the ROUGE metric.
            Changing this argument currently has no effect.
        learning_rate: Learning rate to use for training, defaults to `3e-4`
        enable_ort: Enable Torch ONNX Runtime Optimization: https://onnxruntime.ai/docs/#onnx-runtime-for-training
        n_best_size: The total number of n-best predictions to generate when looking for an answer.
        version_2_with_negative: If true, some of the examples do not have an answer.
        max_answer_length: The maximum length of an answer that can be generated. This is needed because the start and
            end predictions are not conditioned on one another.
        null_score_diff_threshold: The threshold used to select the null answer: if the best answer has a score that is
            less than the score of the null answer minus this threshold, the null answer is selected for this example.
            Only useful when `version_2_with_negative=True`.
        use_stemmer: Whether Porter stemmer should be used to strip word suffixes to improve matching.
    """

    required_extras: str = "text"

    backbones: FlashRegistry = FlashRegistry("backbones") + HUGGINGFACE_BACKBONES

    def __init__(
        self,
        backbone: str = "sshleifer/tiny-distilbert-base-cased-distilled-squad",
        max_source_length: int = 384,
        max_target_length: int = 30,
        padding: Union[str, bool] = "max_length",
        doc_stride: int = 128,
        loss_fn: Optional[Union[Callable, Mapping, Sequence]] = None,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        metrics: METRICS_TYPE = None,
        learning_rate: Optional[float] = None,
        enable_ort: bool = False,
        n_best_size: int = 20,
        version_2_with_negative: bool = True,
        null_score_diff_threshold: float = 0.0,
        use_stemmer: bool = True,
    ):
        self.save_hyperparameters()

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
            output_transform=QuestionAnsweringOutputTransform(),
        )

        self.collate_fn = TextQuestionAnsweringCollate(
            backbone=backbone,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            doc_stride=doc_stride,
            model=self,
        )

        self.model = self.backbones.get(backbone)()
        self.enable_ort = enable_ort
        self.n_best_size = n_best_size
        self.version_2_with_negative = version_2_with_negative
        self.max_target_length = max_target_length
        self.null_score_diff_threshold = null_score_diff_threshold
        self._initialize_model_specific_parameters()

        if _TM_GREATER_EQUAL_0_7_0:
            self.rouge = ROUGEScore(
                use_stemmer=use_stemmer,
            )
        else:
            self.rouge = ROUGEScore(
                True,
                use_stemmer=use_stemmer,
            )

    def _generate_answers(self, pred_start_logits, pred_end_logits, examples):

        all_predictions = collections.OrderedDict()
        if self.version_2_with_negative:
            scores_diff_json = collections.OrderedDict()

        for example_index, example in enumerate(examples):
            min_null_prediction = None
            prelim_predictions = []

            start_logits: Tensor = pred_start_logits[example_index]
            end_logits: Tensor = pred_end_logits[example_index]
            offset_mapping: List[List[int]] = example["offset_mapping"]
            token_is_max_context = example.get("token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes: List[int] = np.argsort(start_logits.clone().detach().cpu().numpy())[
                -1 : -self.n_best_size - 1 : -1
            ].tolist()
            end_indexes: List[int] = np.argsort(end_logits.clone().detach().cpu().numpy())[
                -1 : -self.n_best_size - 1 : -1
            ].tolist()

            max_answer_length = self.max_target_length
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    out_of_bounds_indices = start_index >= len(offset_mapping) or end_index >= len(offset_mapping)
                    unmapped_offsets = offset_mapping[start_index] is None or offset_mapping[end_index] is None
                    if out_of_bounds_indices or unmapped_offsets:
                        continue

                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

            if self.version_2_with_negative:
                # Add the minimum null prediction
                prelim_predictions.append(min_null_prediction)
                null_score = min_null_prediction["score"]

            # Only keep the best `n_best_size` predictions.
            predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[: self.n_best_size]

            # Add back the minimum null prediction if it was removed because of its low score.
            if self.version_2_with_negative and not any(p["offsets"] == (0, 0) for p in predictions):
                predictions.append(min_null_prediction)

            # Use the offsets to gather the answer text in the original context.
            context = example["context"]
            for pred in predictions:
                offsets = pred.pop("offsets")
                pred["text"] = context[offsets[0] : offsets[1]]

            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
                predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

            # Compute the softmax of all scores.
            scores: Tensor = torch.tensor([pred.pop("score") for pred in predictions])
            probs: Tensor = torch.softmax(scores, dim=0)

            # Include the probabilities in our predictions.
            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

            # Pick the best prediction. If the null answer is not possible, this is easy.
            if not self.version_2_with_negative:
                all_predictions[example["example_id"]] = predictions[0]["text"]
            else:
                # Otherwise we first need to find the best non-empty prediction.
                i = 0
                while predictions[i]["text"] == "":
                    i += 1
                best_non_null_pred = predictions[i]
                # Then we compare to the null prediction using the threshold.
                score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
                # To be JSON-serializable.
                scores_diff_json[example["example_id"]] = float(score_diff)
                if score_diff > self.null_score_diff_threshold:
                    all_predictions[example["example_id"]] = ""
                else:
                    all_predictions[example["example_id"]] = best_non_null_pred["text"]

        return all_predictions

    def forward(self, batch: Any) -> Any:
        metadata = batch.pop(DataKeys.METADATA, {})
        outputs = self.model(**batch)
        loss = outputs.loss
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        generated_answers = self._generate_answers(start_logits, end_logits, metadata)
        batch[DataKeys.METADATA] = metadata
        return loss, generated_answers

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def common_step(self, prefix: str, batch: Any) -> Tensor:
        loss, generated_answers = self(batch)
        result = self.compute_metrics(generated_answers, batch[DataKeys.METADATA])
        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(result, on_step=False, on_epoch=True, prog_bar=False)

    def compute_metrics(self, generated_tokens, batch):
        predicted_answers = [generated_tokens[example["example_id"]] for example in batch]
        target_answers = [
            example["answer"]["text"][0] if len(example["answer"]["text"]) > 0 else "" for example in batch
        ]
        return self.rouge(predicted_answers, target_answers)

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        self.common_step("val", batch)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        self.common_step("test", batch)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        _, generated_answers = self(batch)
        return generated_answers

    @property
    def task(self) -> Optional[str]:
        """Override to define AutoConfig task specific parameters stored within the model."""
        return "question_answering"

    def _initialize_model_specific_parameters(self):
        task_specific_params = self.model.config.task_specific_params

        if task_specific_params:
            pars = task_specific_params.get(self.task, {})
            rank_zero_info(f"Overriding model paramameters for {self.task} as defined within the model:\n {pars}")
            self.model.config.update(pars)

    def modules_to_freeze(self) -> Union[Module, Iterable[Union[Module, Iterable]]]:
        """Return the module attributes of the model to be frozen."""
        return self.model.base_model

    def configure_callbacks(self) -> List[Callback]:
        callbacks = super().configure_callbacks() or []
        if self.enable_ort:
            callbacks.append(ORTCallback())
        return callbacks
