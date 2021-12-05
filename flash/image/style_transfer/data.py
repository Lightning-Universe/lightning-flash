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
from typing import Any, Callable, Collection, Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn

from flash.core.data.data_module import DataModule
from flash.core.data.io.input import DataKeys, InputFormat
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.transforms import ApplyToKeys
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE
from flash.core.utilities.stages import RunningStage
from flash.image.classification.data import ImageClassificationFilesInput, ImageClassificationFolderInput
from flash.image.data import ImageFilesInput, ImageNumpyInput, ImageTensorInput

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as T

__all__ = ["StyleTransferInputTransform", "StyleTransferData"]


def _apply_to_input(
    default_transforms_fn, keys: Union[Sequence[DataKeys], DataKeys]
) -> Callable[..., Dict[str, ApplyToKeys]]:
    @functools.wraps(default_transforms_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Optional[Dict[str, ApplyToKeys]]:
        default_transforms = default_transforms_fn(*args, **kwargs)
        if not default_transforms:
            return default_transforms

        return {hook: ApplyToKeys(keys, transform) for hook, transform in default_transforms.items()}

    return wrapper


class StyleTransferInputTransform(InputTransform):
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: int = 256,
    ):
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self.image_size = image_size

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            inputs={
                InputFormat.FILES: ImageFilesInput,
                InputFormat.FOLDERS: ImageClassificationFolderInput,
                InputFormat.NUMPY: ImageNumpyInput,
                InputFormat.TENSORS: ImageTensorInput,
            },
            default_input=InputFormat.FILES,
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {**self.transforms, "image_size": self.image_size}

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(**state_dict)

    @functools.partial(_apply_to_input, keys=DataKeys.INPUT)
    def default_transforms(self) -> Optional[Dict[str, Callable]]:
        if self.training:
            return dict(
                per_sample_transform=T.ToTensor(),
                per_sample_transform_on_device=nn.Sequential(
                    T.Resize(self.image_size),
                    T.CenterCrop(self.image_size),
                ),
            )
        if self.predicting:
            return dict(per_sample_transform=T.Compose([T.Resize(self.image_size), T.ToTensor()]))
        # Style transfer doesn't support a validation or test phase, so we return nothing here
        return None


class StyleTransferData(DataModule):
    input_transform_cls = StyleTransferInputTransform

    @classmethod
    def from_files(
        cls,
        train_files: Optional[Sequence[str]] = None,
        predict_files: Optional[Sequence[str]] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: int = 256,
        **data_module_kwargs: Any,
    ) -> "StyleTransferData":
        return cls(
            ImageFilesInput(RunningStage.TRAINING, train_files),
            predict_dataset=ImageClassificationFilesInput(RunningStage.PREDICTING, predict_files),
            input_transform=cls.input_transform_cls(
                train_transform,
                predict_transform=predict_transform,
                image_size=image_size,
            ),
            **data_module_kwargs,
        )

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: int = 256,
        **data_module_kwargs: Any,
    ) -> "StyleTransferData":
        return cls(
            ImageClassificationFolderInput(RunningStage.TRAINING, train_folder),
            predict_dataset=ImageClassificationFolderInput(RunningStage.PREDICTING, predict_folder),
            input_transform=cls.input_transform_cls(
                train_transform,
                predict_transform=predict_transform,
                image_size=image_size,
            ),
            **data_module_kwargs,
        )

    @classmethod
    def from_numpy(
        cls,
        train_data: Optional[Collection[np.ndarray]] = None,
        predict_data: Optional[Collection[np.ndarray]] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: int = 256,
        **data_module_kwargs: Any,
    ) -> "StyleTransferData":
        return cls(
            ImageNumpyInput(RunningStage.TRAINING, train_data),
            predict_dataset=ImageNumpyInput(RunningStage.PREDICTING, predict_data),
            input_transform=cls.input_transform_cls(
                train_transform,
                predict_transform=predict_transform,
                image_size=image_size,
            ),
            **data_module_kwargs,
        )

    @classmethod
    def from_tensors(
        cls,
        train_data: Optional[Collection[torch.Tensor]] = None,
        predict_data: Optional[Collection[torch.Tensor]] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: int = 256,
        **data_module_kwargs: Any,
    ) -> "StyleTransferData":
        return cls(
            ImageTensorInput(RunningStage.TRAINING, train_data),
            predict_dataset=ImageTensorInput(RunningStage.PREDICTING, predict_data),
            input_transform=cls.input_transform_cls(
                train_transform,
                predict_transform=predict_transform,
                image_size=image_size,
            ),
            **data_module_kwargs,
        )
