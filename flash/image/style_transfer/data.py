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
import functools
import pathlib
from typing import Any, Callable, Dict, Optional, Sequence, Union

from torch import nn

from flash.core.data.data_module import DataModule
from flash.core.data.data_source import DefaultDataKeys, DefaultDataSources
from flash.core.data.process import Preprocess
from flash.core.data.transforms import ApplyToKeys
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE
from flash.image.classification import ImageClassificationData
from flash.image.data import ImageNumpyDataSource, ImagePathsDataSource, ImageTensorDataSource
from flash.image.style_transfer.utils import raise_not_supported

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as T

__all__ = ["StyleTransferPreprocess", "StyleTransferData"]


def _apply_to_input(
    default_transforms_fn, keys: Union[Sequence[DefaultDataKeys], DefaultDataKeys]
) -> Callable[..., Dict[str, ApplyToKeys]]:
    @functools.wraps(default_transforms_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Optional[Dict[str, ApplyToKeys]]:
        default_transforms = default_transforms_fn(*args, **kwargs)
        if not default_transforms:
            return default_transforms

        return {hook: ApplyToKeys(keys, transform) for hook, transform in default_transforms.items()}

    return wrapper


class StyleTransferPreprocess(Preprocess):
    def __init__(
        self,
        train_transform: Optional[Union[Dict[str, Callable]]] = None,
        val_transform: Optional[Union[Dict[str, Callable]]] = None,
        test_transform: Optional[Union[Dict[str, Callable]]] = None,
        predict_transform: Optional[Union[Dict[str, Callable]]] = None,
        image_size: int = 256,
    ):
        if val_transform:
            raise_not_supported("validation")
        if test_transform:
            raise_not_supported("test")

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self.image_size = image_size

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.FILES: ImagePathsDataSource(),
                DefaultDataSources.FOLDERS: ImagePathsDataSource(),
                DefaultDataSources.NUMPY: ImageNumpyDataSource(),
                DefaultDataSources.TENSORS: ImageTensorDataSource(),
                DefaultDataSources.TENSORS: ImageTensorDataSource(),
            },
            default_data_source=DefaultDataSources.FILES,
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {**self.transforms, "image_size": self.image_size}

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(**state_dict)

    @functools.partial(_apply_to_input, keys=DefaultDataKeys.INPUT)
    def default_transforms(self) -> Optional[Dict[str, Callable]]:
        if self.training:
            return dict(
                to_tensor_transform=T.ToTensor(),
                per_sample_transform_on_device=nn.Sequential(
                    T.Resize(self.image_size),
                    T.CenterCrop(self.image_size),
                ),
            )
        if self.predicting:
            return dict(
                pre_tensor_transform=T.Resize(self.image_size),
                to_tensor_transform=T.ToTensor(),
            )
        # Style transfer doesn't support a validation or test phase, so we return nothing here
        return None


class StyleTransferData(ImageClassificationData):
    preprocess_cls = StyleTransferPreprocess

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[Union[str, pathlib.Path]] = None,
        predict_folder: Optional[Union[str, pathlib.Path]] = None,
        train_transform: Optional[Union[str, Dict]] = None,
        predict_transform: Optional[Union[str, Dict]] = None,
        preprocess: Optional[Preprocess] = None,
        **kwargs: Any,
    ) -> "DataModule":

        if any(param in kwargs and kwargs[param] is not None for param in ("val_folder", "val_transform")):
            raise_not_supported("validation")

        if any(param in kwargs and kwargs[param] is not None for param in ("test_folder", "test_transform")):
            raise_not_supported("test")

        preprocess = preprocess or cls.preprocess_cls(
            train_transform=train_transform,
            predict_transform=predict_transform,
        )

        return cls.from_data_source(
            DefaultDataSources.FOLDERS,
            train_data=train_folder,
            predict_data=predict_folder,
            preprocess=preprocess,
            **kwargs,
        )
