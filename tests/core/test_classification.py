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

from flash.core.classification import (
    ClassesOutput,
    FiftyOneLabelsOutput,
    LabelsOutput,
    LogitsOutput,
    ProbabilitiesOutput,
)
from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _CORE_TESTING, _FIFTYONE_AVAILABLE, _IMAGE_AVAILABLE


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_classification_outputs():
    example_output = torch.tensor([-0.1, 0.2, 0.3])  # 3 classes
    labels = ["class_1", "class_2", "class_3"]

    assert torch.allclose(torch.tensor(LogitsOutput().transform(example_output)), example_output)
    assert torch.allclose(
        torch.tensor(ProbabilitiesOutput().transform(example_output)), torch.softmax(example_output, -1)
    )
    assert ClassesOutput().transform(example_output) == 2
    assert LabelsOutput(labels).transform(example_output) == "class_3"


@pytest.mark.skipif(not _CORE_TESTING, reason="Not testing core.")
def test_classification_outputs_multi_label():
    example_output = torch.tensor([-0.1, 0.2, 0.3])  # 3 classes
    labels = ["class_1", "class_2", "class_3"]

    assert torch.allclose(torch.tensor(LogitsOutput(multi_label=True).transform(example_output)), example_output)
    assert torch.allclose(
        torch.tensor(ProbabilitiesOutput(multi_label=True).transform(example_output)),
        torch.sigmoid(example_output),
    )
    assert ClassesOutput(multi_label=True).transform(example_output) == [1, 2]
    assert LabelsOutput(labels, multi_label=True).transform(example_output) == ["class_2", "class_3"]


@pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed.")
@pytest.mark.skipif(not _FIFTYONE_AVAILABLE, reason="fiftyone is not installed for testing")
def test_classification_outputs_fiftyone():
    logits = torch.tensor([-0.1, 0.2, 0.3])
    example_output = {DataKeys.PREDS: logits, DataKeys.METADATA: {"filepath": "something"}}  # 3 classes
    labels = ["class_1", "class_2", "class_3"]

    predictions = FiftyOneLabelsOutput(return_filepath=True).transform(example_output)
    assert predictions["predictions"].label == "2"
    assert predictions["filepath"] == "something"
    predictions = FiftyOneLabelsOutput(labels, return_filepath=True).transform(example_output)
    assert predictions["predictions"].label == "class_3"
    assert predictions["filepath"] == "something"

    predictions = FiftyOneLabelsOutput(store_logits=True, return_filepath=False).transform(example_output)
    assert torch.allclose(torch.tensor(predictions.logits).to(logits.dtype), logits)
    assert torch.allclose(torch.tensor(predictions.confidence), torch.softmax(logits, -1)[-1])
    assert predictions.label == "2"
    predictions = FiftyOneLabelsOutput(labels, store_logits=True, return_filepath=False).transform(example_output)
    assert predictions.label == "class_3"

    predictions = FiftyOneLabelsOutput(store_logits=True, multi_label=True, return_filepath=False).transform(
        example_output
    )
    assert torch.allclose(torch.tensor(predictions.logits).to(logits.dtype), logits)
    assert [c.label for c in predictions.classifications] == ["1", "2"]
    predictions = FiftyOneLabelsOutput(labels, multi_label=True, return_filepath=False).transform(example_output)
    assert [c.label for c in predictions.classifications] == ["class_2", "class_3"]
