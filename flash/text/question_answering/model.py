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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Type, Union

import numpy as np
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_info
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import Metric

from flash.core.data.data_source import DefaultDataKeys
from flash.core.finetuning import FlashBaseFinetuning
from flash.core.model import Task
from flash.core.registry import ExternalRegistry, FlashRegistry
from flash.core.utilities.imports import _TEXT_AVAILABLE
from flash.core.utilities.providers import _HUGGINGFACE
from flash.text.ort_callback import ORTCallback
from flash.text.question_answering.finetuning import QuestionAnsweringFreezeEmbeddings
from flash.text.seq2seq.core.metrics import RougeMetric

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
        loss_fn: Loss function for training.
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        optimizer_kwargs: Additional kwargs to use when creating the optimizer (if not passed as an instance).
        scheduler: The scheduler or scheduler class to use.
        scheduler_kwargs: Additional kwargs to use when creating the scheduler (if not passed as an instance).
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
        rouge_newline_sep: Add a new line at the beginning of each sentence in Rouge Metric calculation.
    """

    required_extras: str = "text"

    backbones: FlashRegistry = FlashRegistry("backbones") + HUGGINGFACE_BACKBONES

    def __init__(
        self,
        backbone: str = "distilbert-base-uncased",
        loss_fn: Optional[Union[Callable, Mapping, Sequence]] = None,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Union[Type[_LRScheduler], str, _LRScheduler]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        metrics: Union[Metric, Callable, Mapping, Sequence, None] = None,
        learning_rate: float = 5e-5,
        enable_ort: bool = False,
        n_best_size: int = 20,
        version_2_with_negative: bool = True,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        use_stemmer: bool = True,
        rouge_newline_sep: bool = True,
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
            learning_rate=learning_rate,
        )
        self.model = self.backbones.get(backbone)()
        self.enable_ort = enable_ort
        self.n_best_size = n_best_size
        self.version_2_with_negative = version_2_with_negative
        self.max_answer_length = max_answer_length
        self.null_score_diff_threshold = null_score_diff_threshold
        self._initialize_model_specific_parameters()

        self.rouge = RougeMetric(
            rouge_newline_sep=rouge_newline_sep,
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

            max_answer_length: int = 30
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
        metadata = batch.pop(DefaultDataKeys.METADATA)
        outputs = self.model(**batch)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        generated_answers = self._generate_answers(start_logits, end_logits, metadata)
        batch[DefaultDataKeys.METADATA] = metadata
        return generated_answers

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def common_step(self, prefix: str, batch: Any) -> torch.Tensor:
        generated_answers = self(batch)
        result = self.compute_metrics(generated_answers, batch[DefaultDataKeys.METADATA])
        self.log_dict(result, on_step=False, on_epoch=True, prog_bar=True)

    def compute_metrics(self, generated_tokens, batch):
        for example in batch:
            predicted_answer = generated_tokens[example["example_id"]]
            target_answer = example["answer"]["text"][0] if len(example["answer"]["text"]) > 0 else ""
            self.rouge.update(predicted_answer, target_answer)

        result = self.rouge.compute()
        return result

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        self.common_step("val", batch)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        self.common_step("test", batch)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        generated_answers = self(batch)
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

    def configure_finetune_callback(self) -> List[FlashBaseFinetuning]:
        return [QuestionAnsweringFreezeEmbeddings(self.model.config.model_type, train_bn=True)]

    def configure_callbacks(self) -> List[Callback]:
        callbacks = super().configure_callbacks() or []
        if self.enable_ort:
            callbacks.append(ORTCallback())
        return callbacks
