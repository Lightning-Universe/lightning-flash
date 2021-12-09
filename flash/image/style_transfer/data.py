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
from typing import Any, Collection, Dict, Optional, Sequence, Type

import numpy as np
import torch

from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.io.input_base import Input
from flash.core.data.new_data_module import DataModule
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE
from flash.image.classification.input import ImageClassificationFilesInput, ImageClassificationFolderInput
from flash.image.data import ImageNumpyInput, ImageTensorInput
from flash.image.style_transfer.input_transform import StyleTransferInputTransform

__all__ = ["StyleTransferInputTransform", "StyleTransferData"]


class StyleTransferData(DataModule):
    input_transform_cls = StyleTransferInputTransform

    @classmethod
    def from_files(
        cls,
        train_files: Optional[Sequence[str]] = None,
        predict_files: Optional[Sequence[str]] = None,
        train_transform: INPUT_TRANSFORM_TYPE = StyleTransferInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = StyleTransferInputTransform,
        input_cls: Type[Input] = ImageClassificationFilesInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any
    ) -> "StyleTransferData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_files, transform=train_transform, **ds_kw),
            predict_input=input_cls(RunningStage.PREDICTING, predict_files, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        train_transform: INPUT_TRANSFORM_TYPE = StyleTransferInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = StyleTransferInputTransform,
        input_cls: Type[Input] = ImageClassificationFolderInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any
    ) -> "StyleTransferData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_folder, transform=train_transform, **ds_kw),
            predict_input=input_cls(RunningStage.PREDICTING, predict_folder, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    @classmethod
    def from_numpy(
        cls,
        train_data: Optional[Collection[np.ndarray]] = None,
        predict_data: Optional[Collection[np.ndarray]] = None,
        train_transform: INPUT_TRANSFORM_TYPE = StyleTransferInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = StyleTransferInputTransform,
        input_cls: Type[Input] = ImageNumpyInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any
    ) -> "StyleTransferData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_data, transform=train_transform, **ds_kw),
            predict_input=input_cls(RunningStage.PREDICTING, predict_data, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    @classmethod
    def from_tensors(
        cls,
        train_data: Optional[Collection[torch.Tensor]] = None,
        predict_data: Optional[Collection[torch.Tensor]] = None,
        train_transform: INPUT_TRANSFORM_TYPE = StyleTransferInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = StyleTransferInputTransform,
        input_cls: Type[Input] = ImageTensorInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any
    ) -> "StyleTransferData":
        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_data, transform=train_transform, **ds_kw),
            predict_input=input_cls(RunningStage.PREDICTING, predict_data, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )
