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
from typing import Any, Dict, Optional, Sequence, Type, Union

import torch
from torch.utils.data import Sampler

from flash.core.data.data_module import DataModule
from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.io.input import Input
from flash.core.data.io.input_transform import INPUT_TRANSFORM_TYPE
from flash.core.integrations.labelstudio.input import _parse_labelstudio_arguments, LabelStudioVideoClassificationInput
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, _PYTORCHVIDEO_AVAILABLE, lazy_import, requires
from flash.core.utilities.stages import RunningStage
from flash.video.classification.input import (
    VideoClassificationFiftyOneInput,
    VideoClassificationFilesInput,
    VideoClassificationFoldersInput,
    VideoClassificationPathsPredictInput,
)
from flash.video.classification.input_transform import VideoClassificationInputTransform

if _FIFTYONE_AVAILABLE:
    SampleCollection = lazy_import("fiftyone.core.collections.SampleCollection")
else:
    SampleCollection = None

if _PYTORCHVIDEO_AVAILABLE:
    from pytorchvideo.data.clip_sampling import ClipSampler
else:
    ClipSampler = None


class VideoClassificationData(DataModule):
    """Data module for Video classification tasks."""

    input_transform_cls = VideoClassificationInputTransform

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
        train_transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
        input_cls: Type[Input] = VideoClassificationFilesInput,
        predict_input_cls: Type[Input] = VideoClassificationFilesInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs,
    ) -> "VideoClassificationData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            video_sampler=video_sampler,
            decode_audio=decode_audio,
            decoder=decoder,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_files, train_targets, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_files, val_targets, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_files, test_targets, transform=test_transform, **ds_kw),
            predict_input_cls(RunningStage.PREDICTING, predict_files, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[str] = None,
        val_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        train_transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
        input_cls: Type[Input] = VideoClassificationFoldersInput,
        predict_input_cls: Type[Input] = VideoClassificationPathsPredictInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs,
    ) -> "VideoClassificationData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            video_sampler=video_sampler,
            decode_audio=decode_audio,
            decoder=decoder,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_folder, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_folder, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_folder, transform=test_transform, **ds_kw),
            predict_input_cls(RunningStage.PREDICTING, predict_folder, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    @classmethod
    @requires("fiftyone")
    def from_fiftyone(
        cls,
        train_dataset: Optional[SampleCollection] = None,
        val_dataset: Optional[SampleCollection] = None,
        test_dataset: Optional[SampleCollection] = None,
        predict_dataset: Optional[SampleCollection] = None,
        train_transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
        label_field: str = "ground_truth",
        input_cls: Type[Input] = VideoClassificationFiftyOneInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs,
    ) -> "VideoClassificationData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            video_sampler=video_sampler,
            decode_audio=decode_audio,
            decoder=decoder,
            label_field=label_field,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_dataset, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_dataset, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_dataset, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_dataset, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    @classmethod
    def from_labelstudio(
        cls,
        export_json: str = None,
        train_export_json: str = None,
        val_export_json: str = None,
        test_export_json: str = None,
        predict_export_json: str = None,
        data_folder: str = None,
        train_data_folder: str = None,
        val_data_folder: str = None,
        test_data_folder: str = None,
        predict_data_folder: str = None,
        train_transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        val_split: Optional[float] = None,
        multi_label: Optional[bool] = False,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
        input_cls: Type[Input] = LabelStudioVideoClassificationInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs,
    ) -> "VideoClassificationData":
        """Creates a :class:`~flash.core.data.data_module.DataModule` object
        from the given export file and data directory using the
        :class:`~flash.core.data.io.input.Input` of name
        :attr:`~flash.core.data.io.input.InputFormat.FOLDERS`
        from the passed or constructed :class:`~flash.core.data.io.input_transform.InputTransform`.

        Args:
            export_json: path to label studio export file
            train_export_json: path to label studio export file for train set,
            overrides export_json if specified
            val_export_json: path to label studio export file for validation
            test_export_json: path to label studio export file for test
            predict_export_json: path to label studio export file for predict
            data_folder: path to label studio data folder
            train_data_folder: path to label studio data folder for train data set,
            overrides data_folder if specified
            val_data_folder: path to label studio data folder for validation data
            test_data_folder: path to label studio data folder for test data
            predict_data_folder: path to label studio data folder for predict data
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            input_transform: The :class:`~flash.core.data.io.input_transform.InputTransform` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.input_transform_cls``
                will be constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            multi_label: Whether the label are multi encoded.
            clip_sampler: Defines how clips should be sampled from each video.
            clip_duration: Defines how long the sampled clips should be for each video.
            clip_sampler_kwargs: Additional keyword arguments to use when constructing the clip sampler.
            video_sampler: Sampler for the internal video container. This defines the order videos are decoded and,
                    if necessary, the distributed split.
            decode_audio: If True, also decode audio from video.
            decoder: Defines what type of decoder used to decode a video.
            data_module_kwargs: Additional keyword arguments to use when constructing the datamodule.

        Returns:
            The constructed data module.

        Examples::

            data_module = DataModule.from_labelstudio(
                export_json='project.json',
                data_folder='label-studio/media/upload',
                val_split=0.8,
            )
        """

        train_data, val_data, test_data, predict_data = _parse_labelstudio_arguments(
            export_json=export_json,
            train_export_json=train_export_json,
            val_export_json=val_export_json,
            test_export_json=test_export_json,
            predict_export_json=predict_export_json,
            data_folder=data_folder,
            train_data_folder=train_data_folder,
            val_data_folder=val_data_folder,
            test_data_folder=test_data_folder,
            predict_data_folder=predict_data_folder,
            val_split=val_split,
            multi_label=multi_label,
        )

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            video_sampler=video_sampler,
            decode_audio=decode_audio,
            decoder=decoder,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_data, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_data, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_data, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )
