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
import glob
import os
from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

from flash import AvailableTasks
from flash.core.data.data_source import (
    DATA_SOURCES_COLLECTION,
    DataSource,
    DataSourcesStore,
    DefaultDataKeys,
    DefaultDataSources,
    LabelsState,
)
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE
from flash.image.data_sources import (
    ImageFiftyOneDataSource,
    ImageNumpyDataSource,
    ImagePathsDataSource,
    ImageTensorDataSource,
)

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets.folder import default_loader


class ImageClassificationDataFrameDataSource(
    DataSource[Tuple[pd.DataFrame, str, Union[str, List[str]], Optional[str]]]
):

    @staticmethod
    def _resolve_file(root: str, file_id: str) -> str:
        if os.path.isabs(file_id):
            pattern = f"{file_id}*"
        else:
            pattern = os.path.join(root, f"*{file_id}*")
        files = glob.glob(pattern)
        if len(files) > 1:
            raise ValueError(
                f"Found multiple matches for pattern: {pattern}. File IDs should uniquely identify the file to load."
            )
        elif len(files) == 0:
            raise ValueError(
                f"Found no matches for pattern: {pattern}. File IDs should uniquely identify the file to load."
            )
        return files[0]

    @staticmethod
    def _resolve_target(label_to_class: Dict[str, int], target_key: str, row: pd.Series) -> pd.Series:
        row[target_key] = label_to_class[row[target_key]]
        return row

    @staticmethod
    def _resolve_multi_target(target_keys: List[str], row: pd.Series) -> pd.Series:
        row[target_keys[0]] = [row[target_key] for target_key in target_keys]
        return row

    def load_data(
        self,
        data: Tuple[pd.DataFrame, str, Union[str, List[str]], Optional[str]],
        dataset: Optional[Any] = None,
    ) -> Sequence[Mapping[str, Any]]:
        data_frame, input_key, target_keys, root = data
        if root is None:
            root = ""

        if not self.predicting:
            if isinstance(target_keys, List):
                dataset.num_classes = len(target_keys)
                self.set_state(LabelsState(target_keys))
                data_frame = data_frame.apply(partial(self._resolve_multi_target, target_keys), axis=1)
                target_keys = target_keys[0]
            else:
                if self.training:
                    labels = list(sorted(data_frame[target_keys].unique()))
                    dataset.num_classes = len(labels)
                    self.set_state(LabelsState(labels))

                labels = self.get_state(LabelsState)

                if labels is not None:
                    labels = labels.labels
                    label_to_class = {v: k for k, v in enumerate(labels)}
                    data_frame = data_frame.apply(partial(self._resolve_target, label_to_class, target_keys), axis=1)

            return [{
                DefaultDataKeys.INPUT: row[input_key],
                DefaultDataKeys.TARGET: row[target_keys],
                DefaultDataKeys.METADATA: dict(root=root),
            } for _, row in data_frame.iterrows()]
        else:
            return [{
                DefaultDataKeys.INPUT: row[input_key],
                DefaultDataKeys.METADATA: dict(root=root),
            } for _, row in data_frame.iterrows()]

    def load_sample(self, sample: Dict[str, Any], dataset: Optional[Any] = None) -> Dict[str, Any]:
        file = self._resolve_file(sample[DefaultDataKeys.METADATA]['root'], sample[DefaultDataKeys.INPUT])
        sample[DefaultDataKeys.INPUT] = default_loader(file)
        return sample


class ImageClassificationCSVDataSource(ImageClassificationDataFrameDataSource):

    def load_data(
        self,
        data: Tuple[str, str, Union[str, List[str]], Optional[str]],
        dataset: Optional[Any] = None,
    ) -> Sequence[Mapping[str, Any]]:
        csv_file, input_key, target_keys, root = data
        data_frame = pd.read_csv(csv_file)
        if root is None:
            root = os.path.dirname(csv_file)
        return super().load_data((data_frame, input_key, target_keys, root), dataset)


ImageClassificationDataSources = {
    DefaultDataSources.FIFTYONE: ImageFiftyOneDataSource,
    DefaultDataSources.FILES: ImagePathsDataSource,
    DefaultDataSources.FOLDERS: ImagePathsDataSource,
    DefaultDataSources.NUMPY: ImageNumpyDataSource,
    DefaultDataSources.TENSORS: ImageTensorDataSource,
    DefaultDataSources.DATA_FRAME: ImageClassificationDataFrameDataSource,
    DefaultDataSources.CSV: ImageClassificationCSVDataSource,
}

ImageClassificationDefaultDataSource = DefaultDataSources.FILES


@DATA_SOURCES_COLLECTION(name=AvailableTasks.ImageClassifier)
def fn():
    return DataSourcesStore(
        data_sources=ImageClassificationDataSources, default_data_source=ImageClassificationDefaultDataSource
    )
