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
from typing import Any, Collection, Dict, Optional, Sequence, Tuple, Type, TYPE_CHECKING

import numpy as np
import torch

from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.io.input_base import Input
from flash.core.data.new_data_module import DataModule
from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, lazy_import
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE
from flash.image.segmentation.input import (
    SemanticSegmentationFiftyOneInput,
    SemanticSegmentationFilesInput,
    SemanticSegmentationFolderInput,
    SemanticSegmentationNumpyInput,
    SemanticSegmentationTensorInput,
)
from flash.image.segmentation.transforms import SemanticSegmentationInputTransform
from flash.image.segmentation.viz import SegmentationMatplotlibVisualization

SampleCollection = None
if _FIFTYONE_AVAILABLE:
    fo = lazy_import("fiftyone")
    if TYPE_CHECKING:
        from fiftyone.core.collections import SampleCollection
else:
    fo = None


class SemanticSegmentationData(DataModule):
    """Data module for semantic segmentation tasks."""

    input_transforms_registry = FlashRegistry("input_transforms")
    input_transform_cls = SemanticSegmentationInputTransform

    @property
    def labels_map(self) -> Optional[Dict[int, Tuple[int, int, int]]]:
        return getattr(self.train_dataset, "labels_map", None)

    @classmethod
    def from_files(
        cls,
        train_files: Optional[Sequence[str]] = None,
        train_targets: Optional[Sequence[str]] = None,
        val_files: Optional[Sequence[str]] = None,
        val_targets: Optional[Sequence[str]] = None,
        test_files: Optional[Sequence[str]] = None,
        test_targets: Optional[Sequence[str]] = None,
        predict_files: Optional[Sequence[str]] = None,
        train_transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        input_cls: Type[Input] = SemanticSegmentationFilesInput,
        num_classes: Optional[int] = None,
        labels_map: Dict[int, Tuple[int, int, int]] = None,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "SemanticSegmentationData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
            num_classes=num_classes,
            labels_map=labels_map,
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
        train_target_folder: Optional[str] = None,
        val_folder: Optional[str] = None,
        val_target_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        test_target_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        train_transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        input_cls: Type[Input] = SemanticSegmentationFolderInput,
        num_classes: Optional[int] = None,
        labels_map: Dict[int, Tuple[int, int, int]] = None,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "SemanticSegmentationData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
            num_classes=num_classes,
            labels_map=labels_map,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_folder, train_target_folder, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_folder, val_target_folder, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_folder, test_target_folder, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_folder, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    @classmethod
    def from_numpy(
        cls,
        train_data: Optional[Collection[np.ndarray]] = None,
        train_targets: Optional[Collection[np.ndarray]] = None,
        val_data: Optional[Collection[np.ndarray]] = None,
        val_targets: Optional[Sequence[np.ndarray]] = None,
        test_data: Optional[Collection[np.ndarray]] = None,
        test_targets: Optional[Sequence[np.ndarray]] = None,
        predict_data: Optional[Collection[np.ndarray]] = None,
        train_transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        input_cls: Type[Input] = SemanticSegmentationNumpyInput,
        num_classes: Optional[int] = None,
        labels_map: Dict[int, Tuple[int, int, int]] = None,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "SemanticSegmentationData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
            num_classes=num_classes,
            labels_map=labels_map,
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
        train_targets: Optional[Collection[torch.Tensor]] = None,
        val_data: Optional[Collection[torch.Tensor]] = None,
        val_targets: Optional[Sequence[torch.Tensor]] = None,
        test_data: Optional[Collection[torch.Tensor]] = None,
        test_targets: Optional[Sequence[torch.Tensor]] = None,
        predict_data: Optional[Collection[torch.Tensor]] = None,
        train_transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        input_cls: Type[Input] = SemanticSegmentationTensorInput,
        num_classes: Optional[int] = None,
        labels_map: Dict[int, Tuple[int, int, int]] = None,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "SemanticSegmentationData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
            num_classes=num_classes,
            labels_map=labels_map,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_data, train_targets, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_data, val_targets, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_data, test_targets, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    @classmethod
    def from_fiftyone(
        cls,
        train_dataset: Optional[SampleCollection] = None,
        val_dataset: Optional[SampleCollection] = None,
        test_dataset: Optional[SampleCollection] = None,
        predict_dataset: Optional[SampleCollection] = None,
        train_transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = SemanticSegmentationInputTransform,
        input_cls: Type[Input] = SemanticSegmentationFiftyOneInput,
        num_classes: Optional[int] = None,
        labels_map: Dict[int, Tuple[int, int, int]] = None,
        transform_kwargs: Optional[Dict] = None,
        label_field: str = "ground_truth",
        **data_module_kwargs: Any,
    ) -> "SemanticSegmentationData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
            label_field=label_field,
            num_classes=num_classes,
            labels_map=labels_map,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_dataset, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_dataset, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_dataset, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_dataset, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    def configure_data_fetcher(self) -> BaseDataFetcher:
        return SegmentationMatplotlibVisualization(labels_map=self.labels_map)

    def set_block_viz_window(self, value: bool) -> None:
        """Setter method to switch on/off matplotlib to pop up windows."""
        self.data_fetcher.block_viz_window = value
