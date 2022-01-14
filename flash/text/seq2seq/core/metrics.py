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
# referenced from
# Library Name: torchtext
# Authors: torchtext authors and @sluks
# Date: 2020-07-18
# Link: https://pytorch.org/text/_modules/torchtext/data/metrics.html#bleu_score
from functools import partial
from typing import Tuple

from deprecate import deprecated, void
from pytorch_lightning.utilities import rank_zero_deprecation
from torchmetrics.text import BLEUScore as _BLEUScore
from torchmetrics.text.rouge import ROUGEScore as _ROUGEScore

_deprecated_text_metrics = partial(deprecated, deprecated_in="0.6.0", remove_in="0.7.0", stream=rank_zero_deprecation)


class BLEUScore(_BLEUScore):
    @_deprecated_text_metrics(target=_BLEUScore)
    def __init__(self, n_gram: int = 4, smooth: bool = False):
        void(n_gram, smooth)


class RougeMetric(_ROUGEScore):
    @_deprecated_text_metrics(target=_ROUGEScore)
    def __init__(
        self,
        newline_sep: bool = False,
        use_stemmer: bool = False,
        rouge_keys: Tuple[str] = ("rouge1", "rouge2", "rougeL", "rougeLsum"),
    ):
        void(newline_sep, use_stemmer, rouge_keys)
