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

from flash.core.classification import Classes, Labels, Logits, Probabilities


def test_classification_serializers():
    example_output = torch.tensor([-0.1, 0.2, 0.3])  # 3 classes

    assert Logits().serialize(example_output) == example_output.tolist()
    assert torch.allclose(torch.tensor(Probabilities().serialize(example_output)), torch.softmax(example_output, -1))
    assert Classes().serialize(example_output) == 2
    assert Labels(['class_1', 'class_2', 'class_3']).serialize(example_output) == 'class_3'
