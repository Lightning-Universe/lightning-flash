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
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from torch.utils.data import Dataset

from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Preprocess
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE
from flash.image.data import ImagePathsDataSource
from flash.image.detection.transforms import default_transforms

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets.folder import default_loader


class FastFaceDataSource(DataSource[Tuple[str, str]]):

    def load_data(self, data: Dataset, dataset: Optional[Any] = None) -> Sequence[Dict[str, Any]]:
        new_data = []
        for img_file_path, targets in zip(data.ids, data.targets):
            new_data.append(
                dict(
                    input=img_file_path,
                    target=dict(
                        boxes=targets["target_boxes"],
                        labels=[1 for _ in range(targets["target_boxes"].shape[0])],
                    )
                )
            )
        return new_data

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        filepath = sample[DefaultDataKeys.INPUT]
        img = default_loader(filepath)
        sample[DefaultDataKeys.INPUT] = img
        w, h = img.size  # WxH
        sample[DefaultDataKeys.METADATA] = {
            "filepath": filepath,
            "size": (h, w),
        }
        return sample


class FaceDetectionPreprocess(Preprocess):

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None
    ):
        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.FILES: ImagePathsDataSource(),
                DefaultDataSources.FOLDERS: ImagePathsDataSource(),
                "fastface": FastFaceDataSource()
            },
            default_data_source=DefaultDataSources.FILES,
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {**self.transforms}

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(**state_dict)

    def default_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_transforms()
