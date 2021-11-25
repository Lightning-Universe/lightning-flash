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
from typing import Any, Dict, List, Optional

from flash.core.data.io.input import LabelsState
from flash.core.data.io.input_base import Input
from flash.core.data.utilities.labels import get_label_details, LabelDetails
from flash.core.data.utilities.samples import format_targets, to_samples


class ClassificationInput(Input):
    def load_data(
        self, inputs: List[Any], targets: Optional[List[Any]] = None, label_details: Optional[LabelDetails] = None
    ) -> List[Dict[str, Any]]:
        samples = to_samples(inputs, targets=targets)
        if targets is not None:
            if self.training:
                label_details = get_label_details(targets)
                self.set_state(LabelsState.from_label_details(label_details))
                self.num_classes = label_details.num_classes
                self.label_details = label_details
            elif label_details is None:
                raise ValueError("In order to format evaluation targets correctly, ``label_details`` must be provided.")

            samples = format_targets(samples, label_details)
        return samples
