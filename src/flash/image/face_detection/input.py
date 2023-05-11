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
from typing import Any, Dict, List

from torch.utils.data import Dataset

from flash.core.data.io.input import DataKeys
from flash.image.data import ImageFilesInput


class FaceDetectionInput(ImageFilesInput):
    """Logic for loading from FDDBDataset."""

    def load_data(self, dataset: Dataset) -> List[Dict[str, Any]]:
        return [
            {
                DataKeys.INPUT: filepath,
                DataKeys.TARGET: targets,
            }
            for filepath, targets in zip(dataset.ids, dataset.targets)
        ]
