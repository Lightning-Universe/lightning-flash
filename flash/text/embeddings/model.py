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
import logging
import os
import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torchmetrics
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_info
from torch import nn, Tensor
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import Metric
from tqdm.autonotebook import trange

from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.process import Serializer
from flash.core.finetuning import FlashBaseFinetuning
from flash.core.model import Task
from flash.core.registry import FlashRegistry
from flash.text.embeddings.backbones import AUTOCONFIG_BACKBONE, AUTOMODEL_BACKBONE, AUTOTOKENIZER_BACKBONE

logger = logging.getLogger(__name__)


class SentenceEmbedder(Task):
    """The ``SentenceEmbedder`` is a :class:`~flash.Task` for generating sentence embeddings, training and
    validation. For more details, see `embeddings`.

    You can change the backbone to any question answering model from `UKPLab/sentence-transformers
    <https://github.com/UKPLab/sentence-transformers>`_ using the ``backbone``
    argument.

    .. note:: When changing the backbone, make sure you pass in the same backbone to the :class:`~flash.Task` and the
        :class:`~flash.core.data.data_module.DataModule` object! Since this is a Sentence Transformers task, make sure you
        use a Sentence Transformers model.

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
    """

    required_extras: str = "text"

    AutoModel_Backbones: FlashRegistry = AUTOMODEL_BACKBONE
    AutoTokenizer_Backbones: FlashRegistry = AUTOTOKENIZER_BACKBONE
    AutoConfig_Backbones: FlashRegistry = AUTOCONFIG_BACKBONE

    def __init__(
        self,
        model_backbone: str = "sentence-transformers/all-MiniLM-L6-v2",
        tokenizer_backbone: str = "sentence-transformers/all-MiniLM-L6-v2",
        enable_ort: bool = False,
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
        # disable HF thousand warnings
        warnings.simplefilter("ignore")
        # set os environ variable for multiprocesses
        os.environ["PYTHONWARNINGS"] = "ignore"
        super().__init__()

        self.config = self.AutoConfig_Backbones.get(model_backbone)
        self.auto_model = self.AutoModel_Backbones.get(model_backbone)
        self.tokenzier = self.AutoTokenizer_Backbones.get(tokenizer_backbone)

        if tokenizer_backbone is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__

    def predict_step(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        """Computes sentence embeddings.

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.
        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = self._target_device

        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.tokenize(sentences_batch)

            with torch.no_grad():
                out_features = self.forward(features)

                if output_value == "token_embeddings":
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features["attention_mask"]):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0 : last_mask_id + 1])
                elif output_value is None:  # Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features["sentence_embedding"])):
                        row = {name: out_features[name][sent_idx] for name in out_features}
                        embeddings.append(row)
                else:  # Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    @property
    def backbone(self):
        return self.model.base_model

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """First call the backbone, then the model head."""

        trans_features = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        if "token_type_ids" in batch:
            trans_features["token_type_ids"] = batch["token_type_ids"]

        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        batch.update({"token_embeddings": output_tokens, "attention_mask": batch["attention_mask"]})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            batch.update({"all_layer_embeddings": hidden_states})

        return batch
