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
from typing import Dict, List, Tuple

import numpy as np
from torch import tensor
from torchmetrics import Metric

from flash.core.utilities.imports import _TEXT_AVAILABLE
from flash.text.seq2seq.summarization.utils import add_newline_to_end_of_each_sentence

if _TEXT_AVAILABLE:
    from rouge_score import rouge_scorer, scoring
    from rouge_score.scoring import AggregateScore, BootstrapAggregator, Score
else:
    AggregateScore, Score, BootstrapAggregator = None, None, object


class RougeMetric(Metric):
    """
    Metric used for automatic summarization. https://www.aclweb.org/anthology/W04-1013/

    Example:

        >>> target = "Is your name John".split()
        >>> preds = "My name is John".split()
        >>> rouge = RougeMetric()   # doctest: +SKIP
        >>> from pprint import pprint
        >>> pprint(rouge(preds, target))  # doctest: +NORMALIZE_WHITESPACE +SKIP
        {'rouge1_fmeasure': 0.25,
         'rouge1_precision': 0.25,
         'rouge1_recall': 0.25,
         'rouge2_fmeasure': 0.0,
         'rouge2_precision': 0.0,
         'rouge2_recall': 0.0,
         'rougeL_fmeasure': 0.25,
         'rougeL_precision': 0.25,
         'rougeL_recall': 0.25,
         'rougeLsum_fmeasure': 0.25,
         'rougeLsum_precision': 0.25,
         'rougeLsum_recall': 0.25}
    """

    def __init__(
        self,
        rouge_newline_sep: bool = False,
        use_stemmer: bool = False,
        rouge_keys: Tuple[str] = ("rouge1", "rouge2", "rougeL", "rougeLsum"),
    ):
        super().__init__()
        if not _TEXT_AVAILABLE:
            raise ModuleNotFoundError("Please, pip install -e '.[text]'")

        self.rouge_newline_sep = rouge_newline_sep
        self.rouge_keys = rouge_keys
        self.use_stemmer = use_stemmer
        self.aggregator = RougeBatchAggregator()
        self.scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=self.use_stemmer)

        for key in rouge_keys:
            self.add_state(key, [])

    def update(self, pred_lns: List[str], tgt_lns: List[str]):
        for pred, tgt in zip(pred_lns, tgt_lns):
            # rougeLsum expects "\n" separated sentences within a summary
            if self.rouge_newline_sep:
                pred = add_newline_to_end_of_each_sentence(pred)
                tgt = add_newline_to_end_of_each_sentence(tgt)
            results = self.scorer.score(pred, tgt)
            for key, score in results.items():
                score = tensor([score.precision, score.recall, score.fmeasure])
                getattr(self, key).append(score)

    def compute(self) -> Dict[str, float]:
        scores = {key: getattr(self, key) for key in self.rouge_keys}
        self.aggregator.add_scores(scores)
        result = self.aggregator.aggregate()
        return format_rouge_results(result)

    def __hash__(self):
        # override to hash list objects.
        # this is a bug in the upstream pytorch release.
        hash_vals = [self.__class__.__name__]

        for key in self._defaults.keys():
            value = getattr(self, key)
            if isinstance(value, list):
                value = tuple(value)
            hash_vals.append(value)

        return hash(tuple(hash_vals))


class RougeBatchAggregator(BootstrapAggregator):
    """
    Aggregates rouge scores and provides confidence intervals.
    """

    def aggregate(self):
        """
        Override function to wrap the final results in `Score` objects.
        This is due to the scores being replaced with a list of torch tensors.
        """
        result = {}
        for score_type, scores in self._scores.items():
            # Stack scores into a 2-d matrix of (sample, measure).
            score_matrix = np.vstack(tuple(scores))
            # Percentiles are returned as (interval, measure).
            percentiles = self._bootstrap_resample(score_matrix)
            # Extract the three intervals (low, mid, high).
            intervals = tuple((Score(*percentiles[j, :]) for j in range(3)))
            result[score_type] = AggregateScore(low=intervals[0], mid=intervals[1], high=intervals[2])
        return result

    def add_scores(self, scores):
        self._scores = scores


def format_rouge_results(result: Dict[str, AggregateScore], decimal_places: int = 4) -> Dict[str, float]:
    flattened_result = {}
    for rouge_key, rouge_aggregate_score in result.items():
        for stat in ["precision", "recall", "fmeasure"]:
            mid = rouge_aggregate_score.mid
            score = round(getattr(mid, stat), decimal_places)
            flattened_result[f"{rouge_key}_{stat}"] = score
    return flattened_result
