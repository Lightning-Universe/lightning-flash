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
import torch

from flash.core.classification import Classes, FiftyOneLabels, Labels, Logits, Probabilities
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE


def test_classification_serializers():
    example_output = torch.tensor([-0.1, 0.2, 0.3])  # 3 classes
    labels = ['class_1', 'class_2', 'class_3']

    assert torch.allclose(torch.tensor(Logits().serialize(example_output)), example_output)
    assert torch.allclose(torch.tensor(FiftyOneLabels(store_logits=True).serialize(example_output).logits), example_output)
    assert torch.allclose(torch.tensor(Probabilities().serialize(example_output)), torch.softmax(example_output, -1))
    assert torch.allclose(torch.tensor(FiftyOneLabels().serialize(example_output).confidence), torch.softmax(example_output, -1)[-1])
    assert Classes().serialize(example_output) == 2
    assert Labels(labels).serialize(example_output) == 'class_3'
    assert FiftyOneLabels(labels).serialize(example_output).label == 'class_3'
    assert FiftyOneLabels().serialize(example_output).label == '2'


def test_classification_serializers_multi_label():
    example_output = torch.tensor([-0.1, 0.2, 0.3])  # 3 classes
    labels = ['class_1', 'class_2', 'class_3']

    assert torch.allclose(torch.tensor(Logits(multi_label=True).serialize(example_output)), example_output)
    assert torch.allclose(torch.tensor(FiftyOneLabels(store_logits=True,multi_label=True).serialize(example_output).logits), example_output)
    assert torch.allclose(
        torch.tensor(Probabilities(multi_label=True).serialize(example_output)),
        torch.sigmoid(example_output),
    )
    assert Classes(multi_label=True).serialize(example_output) == [1, 2]
    assert [c.label for c in FiftyOneLabels(multi_label=True).serialize(example_output).classifications] == ['1', '2']
    assert Labels(labels, multi_label=True).serialize(example_output) == ['class_2', 'class_3']
    assert [c.label for c in FiftyOneLabels(labels, multi_label=True).serialize(example_output).classifications] == ['class_2', 'class_3']


@pytest.mark.skipif(not _FIFTYONE_AVAILABLE, reason="fiftyone is not installed for testing")
def test_classification_serializers_fiftyone():
    example_output = torch.tensor([-0.1, 0.2, 0.3])  # 3 classes
    labels = ['class_1', 'class_2', 'class_3']

    assert torch.allclose(torch.tensor(FiftyOneLabels(store_logits=True).serialize(example_output).logits), example_output)
    assert torch.allclose(torch.tensor(FiftyOneLabels().serialize(example_output).confidence), torch.softmax(example_output, -1)[-1])
    assert FiftyOneLabels(labels).serialize(example_output).label == 'class_3'
    assert FiftyOneLabels().serialize(example_output).label == '2'
    assert torch.allclose(torch.tensor(FiftyOneLabels(store_logits=True,multi_label=True).serialize(example_output).logits), example_output)
    assert [c.label for c in FiftyOneLabels(multi_label=True).serialize(example_output).classifications] == ['1', '2']
    assert [c.label for c in FiftyOneLabels(labels, multi_label=True).serialize(example_output).classifications] == ['class_2', 'class_3']

