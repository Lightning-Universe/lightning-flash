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
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Type, Union

import numpy as np
import pandas as pd
import torch

from flash.audio.classification.input import (
    AudioClassificationCSVInput,
    AudioClassificationDataFrameInput,
    AudioClassificationFilesInput,
    AudioClassificationFolderInput,
    AudioClassificationNumpyInput,
    AudioClassificationTensorInput,
)
from flash.audio.classification.input_transform import AudioClassificationInputTransform
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.input_transform import INPUT_TRANSFORM_TYPE
from flash.core.data.io.input import Input
from flash.core.data.utilities.paths import PATH_TYPE
from flash.core.registry import FlashRegistry
from flash.core.utilities.stages import RunningStage
from flash.image.classification.data import MatplotlibVisualization


class AudioClassificationData(DataModule):
    """Data module for audio classification."""

    input_transform_cls = AudioClassificationInputTransform
    input_transforms_registry = FlashRegistry("input_transforms")

    @classmethod
    def from_files(
        cls,
        train_files: Optional[Sequence[str]] = None,
        train_targets: Optional[Sequence[Any]] = None,
        val_files: Optional[Sequence[str]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_files: Optional[Sequence[str]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_files: Optional[Sequence[str]] = None,
        train_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        input_cls: Type[Input] = AudioClassificationFilesInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "AudioClassificationData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_files, train_targets, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_files, val_targets, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_files, test_targets, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_files, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[str] = None,
        val_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        train_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        input_cls: Type[Input] = AudioClassificationFolderInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "AudioClassificationData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_folder, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_folder, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_folder, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_folder, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    @classmethod
    def from_numpy(
        cls,
        train_data: Optional[Collection[np.ndarray]] = None,
        train_targets: Optional[Collection[Any]] = None,
        val_data: Optional[Collection[np.ndarray]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_data: Optional[Collection[np.ndarray]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_data: Optional[Collection[np.ndarray]] = None,
        train_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        input_cls: Type[Input] = AudioClassificationNumpyInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "AudioClassificationData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_data, train_targets, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_data, val_targets, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_data, test_targets, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    @classmethod
    def from_tensors(
        cls,
        train_data: Optional[Collection[torch.Tensor]] = None,
        train_targets: Optional[Collection[Any]] = None,
        val_data: Optional[Collection[torch.Tensor]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_data: Optional[Collection[torch.Tensor]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_data: Optional[Collection[torch.Tensor]] = None,
        train_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        input_cls: Type[Input] = AudioClassificationTensorInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "AudioClassificationData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_data, train_targets, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_data, val_targets, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_data, test_targets, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    @classmethod
    def from_data_frame(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_data_frame: Optional[pd.DataFrame] = None,
        train_images_root: Optional[str] = None,
        train_resolver: Optional[Callable[[str, str], str]] = None,
        val_data_frame: Optional[pd.DataFrame] = None,
        val_images_root: Optional[str] = None,
        val_resolver: Optional[Callable[[str, str], str]] = None,
        test_data_frame: Optional[pd.DataFrame] = None,
        test_images_root: Optional[str] = None,
        test_resolver: Optional[Callable[[str, str], str]] = None,
        predict_data_frame: Optional[pd.DataFrame] = None,
        predict_images_root: Optional[str] = None,
        predict_resolver: Optional[Callable[[str, str], str]] = None,
        train_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        input_cls: Type[Input] = AudioClassificationDataFrameInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "AudioClassificationData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
        )

        train_data = (train_data_frame, input_field, target_fields, train_images_root, train_resolver)
        val_data = (val_data_frame, input_field, target_fields, val_images_root, val_resolver)
        test_data = (test_data_frame, input_field, target_fields, test_images_root, test_resolver)
        predict_data = (predict_data_frame, input_field, None, predict_images_root, predict_resolver)

        return cls(
            input_cls(RunningStage.TRAINING, *train_data, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, *val_data, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, *test_data, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, *predict_data, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    @classmethod
    def from_csv(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, List[str]]] = None,
        train_file: Optional[PATH_TYPE] = None,
        train_images_root: Optional[PATH_TYPE] = None,
        train_resolver: Optional[Callable[[PATH_TYPE, Any], PATH_TYPE]] = None,
        val_file: Optional[PATH_TYPE] = None,
        val_images_root: Optional[PATH_TYPE] = None,
        val_resolver: Optional[Callable[[PATH_TYPE, Any], PATH_TYPE]] = None,
        test_file: Optional[str] = None,
        test_images_root: Optional[str] = None,
        test_resolver: Optional[Callable[[PATH_TYPE, Any], PATH_TYPE]] = None,
        predict_file: Optional[str] = None,
        predict_images_root: Optional[str] = None,
        predict_resolver: Optional[Callable[[PATH_TYPE, Any], PATH_TYPE]] = None,
        train_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = AudioClassificationInputTransform,
        input_cls: Type[Input] = AudioClassificationCSVInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "AudioClassificationData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
        )

        train_data = (train_file, input_field, target_fields, train_images_root, train_resolver)
        val_data = (val_file, input_field, target_fields, val_images_root, val_resolver)
        test_data = (test_file, input_field, target_fields, test_images_root, test_resolver)
        predict_data = (predict_file, input_field, None, predict_images_root, predict_resolver)

        return cls(
            input_cls(RunningStage.TRAINING, *train_data, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, *val_data, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, *test_data, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, *predict_data, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    def set_block_viz_window(self, value: bool) -> None:
        """Setter method to switch on/off matplotlib to pop up windows."""
        self.data_fetcher.block_viz_window = value

    @staticmethod
    def configure_data_fetcher(*args, **kwargs) -> BaseDataFetcher:
        return MatplotlibVisualization(*args, **kwargs)
