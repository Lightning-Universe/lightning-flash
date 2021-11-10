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
from typing import Any, Dict, List, Tuple

from torch.utils.data._utils.collate import default_collate

from flash.core.data.io.input import DataKeys


def convert_predictions(predictions: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List]:
    # Flatten list if batches were used
    if all(isinstance(fl, list) for fl in predictions):
        unrolled_predictions = []
        for prediction_batch in predictions:
            unrolled_predictions.extend(prediction_batch)
        predictions = unrolled_predictions
    result = default_collate(predictions)
    inputs = result.pop(DataKeys.INPUT)
    return result, inputs
