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
import pytest
import torch

from flash.text.seq2seq.core.metrics import BLEUScore, RougeMetric
from tests.helpers.utils import _TEXT_TESTING


@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_rouge():
    preds = "My name is John".split()
    target = "Is your name John".split()
    metric = RougeMetric()
    assert torch.allclose(torch.tensor(metric(preds, target)["rouge1_recall"]).float(), torch.tensor(0.25), 1e-4)


@pytest.mark.parametrize("smooth, expected", [(False, 0.7598), (True, 0.8091)])
def test_bleu_score(smooth, expected):
    translate_corpus = ["the cat is on the mat".split()]
    reference_corpus = [["there is a cat on the mat".split(), "a cat is on the mat".split()]]
    metric = BLEUScore(smooth=smooth)
    assert torch.allclose(metric(translate_corpus, reference_corpus), torch.tensor(expected), 1e-4)
