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

import pandas as pd
import torch
from torch.utils.data import Sampler

from flash.core.data.data_module import DataModule
from flash.core.data.io.input import Input
from flash.core.data.io.input_transform import INPUT_TRANSFORM_TYPE
from flash.core.data.utilities.classification import TargetFormatter
from flash.core.data.utilities.paths import PATH_TYPE
from flash.core.integrations.labelstudio.input import _parse_labelstudio_arguments, LabelStudioVideoClassificationInput
from flash.core.utilities.imports import (
    _FIFTYONE_AVAILABLE,
    _PYTORCHVIDEO_AVAILABLE,
    _VIDEO_EXTRAS_TESTING,
    _VIDEO_TESTING,
    requires,
)
from flash.core.utilities.stages import RunningStage
from flash.video.classification.input import (
    VideoClassificationCSVInput,
    VideoClassificationCSVPredictInput,
    VideoClassificationDataFrameInput,
    VideoClassificationDataFramePredictInput,
    VideoClassificationFiftyOneInput,
    VideoClassificationFiftyOnePredictInput,
    VideoClassificationFilesInput,
    VideoClassificationFoldersInput,
    VideoClassificationPathsPredictInput,
    VideoClassificationTensorsInput,
    VideoClassificationTensorsPredictInput,
)
from flash.video.classification.input_transform import VideoClassificationInputTransform

if _FIFTYONE_AVAILABLE:
    SampleCollection = "fiftyone.core.collections.SampleCollection"
else:
    SampleCollection = None

if _PYTORCHVIDEO_AVAILABLE:
    from pytorchvideo.data.clip_sampling import ClipSampler
else:
    ClipSampler = None

# Skip doctests if requirements aren't available
__doctest_skip__ = []
if not _VIDEO_TESTING:
    __doctest_skip__ += [
        "VideoClassificationData",
        "VideoClassificationData.from_files",
        "VideoClassificationData.from_folders",
        "VideoClassificationData.from_data_frame",
        "VideoClassificationData.from_csv",
        "VideoClassificationData.from_tensors",
    ]
if not _VIDEO_EXTRAS_TESTING:
    __doctest_skip__ += ["VideoClassificationData.from_fiftyone"]


class VideoClassificationData(DataModule):
    """The ``VideoClassificationData`` class is a :class:`~flash.core.data.data_module.DataModule` with a set of
    classmethods for loading data for video classification."""

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
        target_formatter: Optional[TargetFormatter] = None,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
        input_cls: Type[Input] = VideoClassificationFilesInput,
        predict_input_cls: Type[Input] = VideoClassificationPathsPredictInput,
        transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs,
    ) -> "VideoClassificationData":
        """Load the :class:`~flash.video.classification.data.VideoClassificationData` from lists of files and
        corresponding lists of targets.

        The supported file extensions are: ``.mp4``, and ``.avi``.
        The targets can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_files: The list of video files to use when training.
            train_targets: The list of targets to use when training.
            val_files: The list of video files to use when validating.
            val_targets: The list of targets to use when validating.
            test_files: The list of video files to use when testing.
            test_targets: The list of targets to use when testing.
            predict_files: The list of video files to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            clip_sampler: The clip sampler to use. One of: ``"uniform"``, ``"random"``, ``"constant_clips_per_video"``.
            clip_duration: The duration of clips to sample.
            clip_sampler_kwargs: Additional keyword arguments to use when constructing the clip sampler.
            video_sampler: Sampler for the internal video container. This defines the order videos are decoded and,
                if necessary, the distributed split.
            decode_audio: If True, also decode audio from video.
            decoder: The decoder to use to decode videos. One of: ``"pyav"``, ``"torchvision"``. Not used for frame
                videos.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            predict_input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the prediction data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.video.classification.data.VideoClassificationData`.

        Examples
        ________

        .. testsetup::

            >>> import torch
            >>> from torchvision import io
            >>> data = torch.randint(255, (10, 64, 64, 3))
            >>> _ = [io.write_video(f"video_{i}.mp4", data, 5, "libx264rgb", {"crf": "0"}) for i in range(1, 4)]
            >>> _ = [io.write_video(f"predict_video_{i}.mp4", data, 5, "libx264rgb", {"crf": "0"}) for i in range(1, 4)]

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.video import VideoClassifier, VideoClassificationData
            >>> datamodule = VideoClassificationData.from_files(
            ...     train_files=["video_1.mp4", "video_2.mp4", "video_3.mp4"],
            ...     train_targets=["cat", "dog", "cat"],
            ...     predict_files=["predict_video_1.mp4", "predict_video_2.mp4", "predict_video_3.mp4"],
            ...     transform_kwargs=dict(image_size=(244, 244)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = VideoClassifier(backbone="x3d_xs", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import os
            >>> _ = [os.remove(f"video_{i}.mp4") for i in range(1, 4)]
            >>> _ = [os.remove(f"predict_video_{i}.mp4") for i in range(1, 4)]
        """
        ds_kw = dict(
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            decode_audio=decode_audio,
            decoder=decoder,
        )

        train_input = input_cls(
            RunningStage.TRAINING,
            train_files,
            train_targets,
            video_sampler=video_sampler,
            target_formatter=target_formatter,
            **ds_kw,
        )
        target_formatter = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(
                RunningStage.VALIDATING,
                val_files,
                val_targets,
                video_sampler=video_sampler,
                target_formatter=target_formatter,
                **ds_kw,
            ),
            input_cls(
                RunningStage.TESTING,
                test_files,
                test_targets,
                video_sampler=video_sampler,
                target_formatter=target_formatter,
                **ds_kw,
            ),
            predict_input_cls(RunningStage.PREDICTING, predict_files, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[str] = None,
        val_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        target_formatter: Optional[TargetFormatter] = None,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
        input_cls: Type[Input] = VideoClassificationFoldersInput,
        predict_input_cls: Type[Input] = VideoClassificationPathsPredictInput,
        transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs,
    ) -> "VideoClassificationData":
        """Load the :class:`~flash.video.classification.data.VideoClassificationData` from folders containing
        videos.

        The supported file extensions are: ``.mp4``, and ``.avi``.
        For train, test, and validation data, the folders are expected to contain a sub-folder for each class.
        Here's the required structure:

        .. code-block::

            train_folder
            ├── cat
            │   ├── video_1.mp4
            │   ├── video_3.mp4
            │   ...
            └── dog
                ├── video_2.mp4
                ...

        For prediction, the folder is expected to contain the files for inference, like this:

        .. code-block::

            predict_folder
            ├── predict_video_1.mp4
            ├── predict_video_2.mp4
            ├── predict_video_3.mp4
            ...

        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_folder: The folder containing videos to use when training.
            val_folder: The folder containing videos to use when validating.
            test_folder: The folder containing videos to use when testing.
            predict_folder: The folder containing videos to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            clip_sampler: The clip sampler to use. One of: ``"uniform"``, ``"random"``, ``"constant_clips_per_video"``.
            clip_duration: The duration of clips to sample.
            clip_sampler_kwargs: Additional keyword arguments to use when constructing the clip sampler.
            video_sampler: Sampler for the internal video container. This defines the order videos are decoded and,
                if necessary, the distributed split.
            decode_audio: If True, also decode audio from video.
            decoder: The decoder to use to decode videos. One of: ``"pyav"``, ``"torchvision"``. Not used for frame
                videos.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            predict_input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the prediction data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.video.classification.data.VideoClassificationData`.

        Examples
        ________

        .. testsetup::

            >>> import os
            >>> import torch
            >>> from torchvision import io
            >>> data = torch.randint(255, (10, 64, 64, 3))
            >>> os.makedirs(os.path.join("train_folder", "cat"), exist_ok=True)
            >>> os.makedirs(os.path.join("train_folder", "dog"), exist_ok=True)
            >>> os.makedirs("predict_folder", exist_ok=True)
            >>> _ = [io.write_video(
            ...     os.path.join("train_folder", label, f"video_{i + 1}.mp4"), data, 5, "libx264rgb", {"crf": "0"}
            ... ) for i, label in enumerate(["cat", "dog", "cat"])]
            >>> _ = [
            ...     io.write_video(
            ...         os.path.join("predict_folder", f"predict_video_{i}.mp4"), data, 5, "libx264rgb", {"crf": "0"}
            ...     ) for i in range(1, 4)
            ... ]

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.video import VideoClassifier, VideoClassificationData
            >>> datamodule = VideoClassificationData.from_folders(
            ...     train_folder="train_folder",
            ...     predict_folder="predict_folder",
            ...     transform_kwargs=dict(image_size=(244, 244)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = VideoClassifier(backbone="x3d_xs", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import shutil
            >>> shutil.rmtree("train_folder")
            >>> shutil.rmtree("predict_folder")
        """
        ds_kw = dict(
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            decode_audio=decode_audio,
            decoder=decoder,
        )

        train_input = input_cls(
            RunningStage.TRAINING,
            train_folder,
            video_sampler=video_sampler,
            target_formatter=target_formatter,
            **ds_kw,
        )
        target_formatter = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(
                RunningStage.VALIDATING,
                val_folder,
                video_sampler=video_sampler,
                target_formatter=target_formatter,
                **ds_kw,
            ),
            input_cls(
                RunningStage.TESTING,
                test_folder,
                video_sampler=video_sampler,
                target_formatter=target_formatter,
                **ds_kw,
            ),
            predict_input_cls(RunningStage.PREDICTING, predict_folder, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_data_frame(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, Sequence[str]]] = None,
        train_data_frame: Optional[pd.DataFrame] = None,
        train_videos_root: Optional[str] = None,
        train_resolver: Optional[Callable[[str, str], str]] = None,
        val_data_frame: Optional[pd.DataFrame] = None,
        val_videos_root: Optional[str] = None,
        val_resolver: Optional[Callable[[str, str], str]] = None,
        test_data_frame: Optional[pd.DataFrame] = None,
        test_videos_root: Optional[str] = None,
        test_resolver: Optional[Callable[[str, str], str]] = None,
        predict_data_frame: Optional[pd.DataFrame] = None,
        predict_videos_root: Optional[str] = None,
        predict_resolver: Optional[Callable[[str, str], str]] = None,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
        input_cls: Type[Input] = VideoClassificationDataFrameInput,
        predict_input_cls: Type[Input] = VideoClassificationDataFramePredictInput,
        target_formatter: Optional[TargetFormatter] = None,
        transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "VideoClassificationData":
        """Load the :class:`~flash.video.classification.data.VideoClassificationData` from pandas DataFrame objects
        containing video file paths and their corresponding targets.

        Input video file paths will be extracted from the ``input_field`` in the DataFrame.
        The supported file extensions are: ``.mp4``, and ``.avi``.
        The targets will be extracted from the ``target_fields`` in the DataFrame and can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            input_field: The field (column name) in the DataFrames containing the video file paths.
            target_fields: The field (column name) or list of fields in the DataFrames containing the targets.
            train_data_frame: The DataFrame to use when training.
            train_videos_root: The root directory containing train videos.
            train_resolver: Optionally provide a function which converts an entry from the ``input_field`` into a video
                file path.
            val_data_frame: The DataFrame to use when validating.
            val_videos_root: The root directory containing validation videos.
            val_resolver: Optionally provide a function which converts an entry from the ``input_field`` into a video
                file path.
            test_data_frame: The DataFrame to use when testing.
            test_videos_root: The root directory containing test videos.
            test_resolver: Optionally provide a function which converts an entry from the ``input_field`` into a video
                file path.
            predict_data_frame: The DataFrame to use when predicting.
            predict_videos_root: The root directory containing predict videos.
            predict_resolver: Optionally provide a function which converts an entry from the ``input_field`` into a
                video file path.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            clip_sampler: The clip sampler to use. One of: ``"uniform"``, ``"random"``, ``"constant_clips_per_video"``.
            clip_duration: The duration of clips to sample.
            clip_sampler_kwargs: Additional keyword arguments to use when constructing the clip sampler.
            video_sampler: Sampler for the internal video container. This defines the order videos are decoded and,
                if necessary, the distributed split.
            decode_audio: If True, also decode audio from video.
            decoder: The decoder to use to decode videos. One of: ``"pyav"``, ``"torchvision"``. Not used for frame
                videos.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            predict_input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the prediction data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.video.classification.data.VideoClassificationData`.

        Examples
        ________

        .. testsetup::

            >>> import os
            >>> import torch
            >>> from torchvision import io
            >>> data = torch.randint(255, (10, 64, 64, 3))
            >>> os.makedirs("train_folder", exist_ok=True)
            >>> os.makedirs("predict_folder", exist_ok=True)
            >>> _ = [io.write_video(
            ...     os.path.join("train_folder", f"video_{i}.mp4"), data, 5, "libx264rgb", {"crf": "0"}
            ... ) for i in range(1, 4)]
            >>> _ = [
            ...     io.write_video(
            ...         os.path.join("predict_folder", f"predict_video_{i}.mp4"), data, 5, "libx264rgb", {"crf": "0"}
            ...     ) for i in range(1, 4)
            ... ]

        .. doctest::

            >>> from pandas import DataFrame
            >>> from flash import Trainer
            >>> from flash.video import VideoClassifier, VideoClassificationData
            >>> train_data_frame = DataFrame.from_dict(
            ...     {
            ...         "videos": ["video_1.mp4", "video_2.mp4", "video_3.mp4"],
            ...         "targets": ["cat", "dog", "cat"],
            ...     }
            ... )
            >>> predict_data_frame = DataFrame.from_dict(
            ...     {
            ...         "videos": ["predict_video_1.mp4", "predict_video_2.mp4", "predict_video_3.mp4"],
            ...     }
            ... )
            >>> datamodule = VideoClassificationData.from_data_frame(
            ...     "videos",
            ...     "targets",
            ...     train_data_frame=train_data_frame,
            ...     train_videos_root="train_folder",
            ...     predict_data_frame=predict_data_frame,
            ...     predict_videos_root="predict_folder",
            ...     transform_kwargs=dict(image_size=(244, 244)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = VideoClassifier(backbone="x3d_xs", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import shutil
            >>> shutil.rmtree("train_folder")
            >>> shutil.rmtree("predict_folder")
            >>> del train_data_frame
            >>> del predict_data_frame
        """
        ds_kw = dict(
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            decode_audio=decode_audio,
            decoder=decoder,
        )

        train_data = (train_data_frame, input_field, target_fields, train_videos_root, train_resolver)
        val_data = (val_data_frame, input_field, target_fields, val_videos_root, val_resolver)
        test_data = (test_data_frame, input_field, target_fields, test_videos_root, test_resolver)
        predict_data = (predict_data_frame, input_field, predict_videos_root, predict_resolver)

        train_input = input_cls(
            RunningStage.TRAINING,
            *train_data,
            video_sampler=video_sampler,
            target_formatter=target_formatter,
            **ds_kw,
        )
        target_formatter = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(
                RunningStage.VALIDATING,
                *val_data,
                video_sampler=video_sampler,
                target_formatter=target_formatter,
                **ds_kw,
            ),
            input_cls(
                RunningStage.TESTING,
                *test_data,
                video_sampler=video_sampler,
                target_formatter=target_formatter,
                **ds_kw,
            ),
            predict_input_cls(RunningStage.PREDICTING, *predict_data, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_tensors(
        cls,
        train_data: Optional[Union[Collection[torch.Tensor], torch.Tensor]] = None,
        train_targets: Optional[Collection[Any]] = None,
        val_data: Optional[Union[Collection[torch.Tensor], torch.Tensor]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_data: Optional[Collection[torch.Tensor]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_data: Optional[Union[Collection[torch.Tensor], torch.Tensor]] = None,
        target_formatter: Optional[TargetFormatter] = None,
        video_sampler: Type[Sampler] = torch.utils.data.SequentialSampler,
        input_cls: Type[Input] = VideoClassificationTensorsInput,
        predict_input_cls: Type[Input] = VideoClassificationTensorsPredictInput,
        transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "VideoClassificationData":
        """Load the :class:`~flash.video.classification.data.VideoClassificationData` from a dictionary containing
        PyTorch tensors representing input video frames and their corresponding targets.

        Input tensor(s) will be extracted from the ``input_field`` in the ``dict``.
        The targets will be extracted from the ``target_fields`` in the ``dict`` and can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.

        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_data: The torch tensor or list of tensors to use when training.
            train_targets: The list of targets to use when training.
            val_data: The torch tensor or list of tensors to use when validating.
            val_targets: The list of targets to use when validating.
            test_data: The torch tensor or list of tensors to use when testing.
            test_targets: The list of targets to use when testing.
            predict_data: The torch tensor or list of tensors to use when predicting.
            train_data: A torch tensor or list of tensors to use when training.
            train_targets: The list of targets to use when training.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            video_sampler: Sampler for the internal video container. This defines the order tensors are used and,
                if necessary, the distributed split.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            predict_input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the prediction data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.video.classification.data.VideoClassificationData`.

        Examples
        ________

        .. doctest::

            >>> import torch
            >>> from flash import Trainer
            >>> from flash.video import VideoClassifier, VideoClassificationData
            >>> frame = torch.randint(low=0, high=255, size=(3, 5, 10, 10), dtype=torch.uint8, device="cpu")
            >>> datamodule = VideoClassificationData.from_tensors(
            ...     train_data=[frame, frame, frame],
            ...     train_targets=["fruit", "vegetable", "fruit"],
            ...     val_data=[frame, frame],
            ...     val_targets=["vegetable", "fruit"],
            ...     predict_data=[frame],
            ...     batch_size=1,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['fruit', 'vegetable']
            >>> model = VideoClassifier(backbone="x3d_xs", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> del frame
        """

        train_input = input_cls(
            RunningStage.TRAINING,
            train_data,
            train_targets,
            video_sampler=video_sampler,
            target_formatter=target_formatter,
        )
        target_formatter = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(
                RunningStage.VALIDATING,
                val_data,
                val_targets,
                video_sampler=video_sampler,
                target_formatter=target_formatter,
            ),
            input_cls(
                RunningStage.TESTING,
                test_data,
                test_targets,
                video_sampler=video_sampler,
                target_formatter=target_formatter,
            ),
            predict_input_cls(RunningStage.PREDICTING, predict_data),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_csv(
        cls,
        input_field: str,
        target_fields: Optional[Union[str, List[str]]] = None,
        train_file: Optional[PATH_TYPE] = None,
        train_videos_root: Optional[PATH_TYPE] = None,
        train_resolver: Optional[Callable[[PATH_TYPE, Any], PATH_TYPE]] = None,
        val_file: Optional[PATH_TYPE] = None,
        val_videos_root: Optional[PATH_TYPE] = None,
        val_resolver: Optional[Callable[[PATH_TYPE, Any], PATH_TYPE]] = None,
        test_file: Optional[str] = None,
        test_videos_root: Optional[str] = None,
        test_resolver: Optional[Callable[[PATH_TYPE, Any], PATH_TYPE]] = None,
        predict_file: Optional[str] = None,
        predict_videos_root: Optional[str] = None,
        predict_resolver: Optional[Callable[[PATH_TYPE, Any], PATH_TYPE]] = None,
        target_formatter: Optional[TargetFormatter] = None,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
        input_cls: Type[Input] = VideoClassificationCSVInput,
        predict_input_cls: Type[Input] = VideoClassificationCSVPredictInput,
        transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "VideoClassificationData":
        """Load the :class:`~flash.video.classification.data.VideoClassificationData` from CSV files containing
        video file paths and their corresponding targets.

        Input video file paths will be extracted from the ``input_field`` column in the CSV files.
        The supported file extensions are: ``.mp4``, and ``.avi``.
        The targets will be extracted from the ``target_fields`` in the CSV files and can be in any of our
        :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            input_field: The field (column name) in the CSV files containing the video file paths.
            target_fields: The field (column name) or list of fields in the CSV files containing the targets.
            train_file: The CSV file to use when training.
            train_videos_root: The root directory containing train videos.
            train_resolver: Optionally provide a function which converts an entry from the ``input_field`` into a video
                file path.
            val_file: The CSV file to use when validating.
            val_videos_root: The root directory containing validation videos.
            val_resolver: Optionally provide a function which converts an entry from the ``input_field`` into a video
                file path.
            test_file: The CSV file to use when testing.
            test_videos_root: The root directory containing test videos.
            test_resolver: Optionally provide a function which converts an entry from the ``input_field`` into a video
                file path.
            predict_file: The CSV file to use when predicting.
            predict_videos_root: The root directory containing predict videos.
            predict_resolver: Optionally provide a function which converts an entry from the ``input_field`` into a
                video file path.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            clip_sampler: The clip sampler to use. One of: ``"uniform"``, ``"random"``, ``"constant_clips_per_video"``.
            clip_duration: The duration of clips to sample.
            clip_sampler_kwargs: Additional keyword arguments to use when constructing the clip sampler.
            video_sampler: Sampler for the internal video container. This defines the order videos are decoded and,
                if necessary, the distributed split.
            decode_audio: If True, also decode audio from video.
            decoder: The decoder to use to decode videos. One of: ``"pyav"``, ``"torchvision"``. Not used for frame
                videos.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            predict_input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the prediction data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.video.classification.data.VideoClassificationData`.

        Examples
        ________

        The files can be in Comma Separated Values (CSV) format with either a ``.csv`` or ``.txt`` extension.

        .. testsetup::

            >>> import os
            >>> import torch
            >>> from torchvision import io
            >>> from pandas import DataFrame
            >>> data = torch.randint(255, (10, 64, 64, 3))
            >>> os.makedirs("train_folder", exist_ok=True)
            >>> os.makedirs("predict_folder", exist_ok=True)
            >>> _ = [io.write_video(
            ...     os.path.join("train_folder", f"video_{i}.mp4"), data, 5, "libx264rgb", {"crf": "0"}
            ... ) for i in range(1, 4)]
            >>> _ = [
            ...     io.write_video(
            ...         os.path.join("predict_folder", f"predict_video_{i}.mp4"), data, 5, "libx264rgb", {"crf": "0"}
            ...     ) for i in range(1, 4)
            ... ]
            >>> DataFrame.from_dict({
            ...         "videos": ["video_1.mp4", "video_2.mp4", "video_3.mp4"],
            ...         "targets": ["cat", "dog", "cat"],
            ... }).to_csv("train_data.csv", index=False)
            >>> DataFrame.from_dict({
            ...         "videos": ["predict_video_1.mp4", "predict_video_2.mp4", "predict_video_3.mp4"],
            ... }).to_csv("predict_data.csv", index=False)

        The file ``train_data.csv`` contains the following:

        .. code-block::

            videos,targets
            video_1.mp4,cat
            video_2.mp4,dog
            video_3.mp4,cat

        The file ``predict_data.csv`` contains the following:

        .. code-block::

            videos
            predict_video_1.mp4
            predict_video_2.mp4
            predict_video_3.mp4

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.video import VideoClassifier, VideoClassificationData
            >>> datamodule = VideoClassificationData.from_csv(
            ...     "videos",
            ...     "targets",
            ...     train_file="train_data.csv",
            ...     train_videos_root="train_folder",
            ...     predict_file="predict_data.csv",
            ...     predict_videos_root="predict_folder",
            ...     transform_kwargs=dict(image_size=(244, 244)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = VideoClassifier(backbone="x3d_xs", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import shutil
            >>> shutil.rmtree("train_folder")
            >>> shutil.rmtree("predict_folder")
            >>> os.remove("train_data.csv")
            >>> os.remove("predict_data.csv")

        Alternatively, the files can be in Tab Separated Values (TSV) format with a ``.tsv`` extension.

        .. testsetup::

            >>> import os
            >>> import torch
            >>> from torchvision import io
            >>> from pandas import DataFrame
            >>> data = torch.randint(255, (10, 64, 64, 3))
            >>> os.makedirs("train_folder", exist_ok=True)
            >>> os.makedirs("predict_folder", exist_ok=True)
            >>> _ = [io.write_video(
            ...     os.path.join("train_folder", f"video_{i}.mp4"), data, 5, "libx264rgb", {"crf": "0"}
            ... ) for i in range(1, 4)]
            >>> _ = [
            ...     io.write_video(
            ...         os.path.join("predict_folder", f"predict_video_{i}.mp4"), data, 5, "libx264rgb", {"crf": "0"}
            ...     ) for i in range(1, 4)
            ... ]
            >>> DataFrame.from_dict({
            ...         "videos": ["video_1.mp4", "video_2.mp4", "video_3.mp4"],
            ...         "targets": ["cat", "dog", "cat"],
            ... }).to_csv("train_data.tsv", sep="\\t", index=False)
            >>> DataFrame.from_dict({
            ...         "videos": ["predict_video_1.mp4", "predict_video_2.mp4", "predict_video_3.mp4"],
            ... }).to_csv("predict_data.tsv", sep="\\t", index=False)

        The file ``train_data.tsv`` contains the following:

        .. code-block::

            videos      targets
            video_1.mp4 cat
            video_2.mp4 dog
            video_3.mp4 cat

        The file ``predict_data.tsv`` contains the following:

        .. code-block::

            videos
            predict_video_1.mp4
            predict_video_2.mp4
            predict_video_3.mp4

        .. doctest::

            >>> from flash import Trainer
            >>> from flash.video import VideoClassifier, VideoClassificationData
            >>> datamodule = VideoClassificationData.from_csv(
            ...     "videos",
            ...     "targets",
            ...     train_file="train_data.tsv",
            ...     train_videos_root="train_folder",
            ...     predict_file="predict_data.tsv",
            ...     predict_videos_root="predict_folder",
            ...     transform_kwargs=dict(image_size=(244, 244)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = VideoClassifier(backbone="x3d_xs", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import shutil
            >>> shutil.rmtree("train_folder")
            >>> shutil.rmtree("predict_folder")
            >>> os.remove("train_data.tsv")
            >>> os.remove("predict_data.tsv")
        """
        ds_kw = dict(
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            decode_audio=decode_audio,
            decoder=decoder,
        )

        train_data = (train_file, input_field, target_fields, train_videos_root, train_resolver)
        val_data = (val_file, input_field, target_fields, val_videos_root, val_resolver)
        test_data = (test_file, input_field, target_fields, test_videos_root, test_resolver)
        predict_data = (predict_file, input_field, predict_videos_root, predict_resolver)

        train_input = input_cls(
            RunningStage.TRAINING,
            *train_data,
            video_sampler=video_sampler,
            target_formatter=target_formatter,
            **ds_kw,
        )
        target_formatter = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(
                RunningStage.VALIDATING,
                *val_data,
                video_sampler=video_sampler,
                target_formatter=target_formatter,
                **ds_kw,
            ),
            input_cls(
                RunningStage.TESTING,
                *test_data,
                video_sampler=video_sampler,
                target_formatter=target_formatter,
                **ds_kw,
            ),
            predict_input_cls(RunningStage.PREDICTING, *predict_data, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
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
        target_formatter: Optional[TargetFormatter] = None,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
        label_field: str = "ground_truth",
        input_cls: Type[Input] = VideoClassificationFiftyOneInput,
        predict_input_cls: Type[Input] = VideoClassificationFiftyOnePredictInput,
        transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs,
    ) -> "VideoClassificationData":
        """Load the :class:`~flash.video.classification.data.VideoClassificationData` from FiftyOne
        ``SampleCollection`` objects.

        The supported file extensions are: ``.mp4``, and ``.avi``.
        The targets will be extracted from the ``label_field`` in the ``SampleCollection`` objects and can be in any
        of our :ref:`supported classification target formats <formatting_classification_targets>`.
        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_dataset: The ``SampleCollection`` to use when training.
            val_dataset: The ``SampleCollection`` to use when validating.
            test_dataset: The ``SampleCollection`` to use when testing.
            predict_dataset: The ``SampleCollection`` to use when predicting.
            label_field: The field in the ``SampleCollection`` objects containing the targets.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. See :ref:`formatting_classification_targets` for more details.
            clip_sampler: The clip sampler to use. One of: ``"uniform"``, ``"random"``, ``"constant_clips_per_video"``.
            clip_duration: The duration of clips to sample.
            clip_sampler_kwargs: Additional keyword arguments to use when constructing the clip sampler.
            video_sampler: Sampler for the internal video container. This defines the order videos are decoded and,
                if necessary, the distributed split.
            decode_audio: If True, also decode audio from video.
            decoder: The decoder to use to decode videos. One of: ``"pyav"``, ``"torchvision"``. Not used for frame
                videos.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            predict_input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the prediction data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.video.classification.data.VideoClassificationData`.

        Examples
        ________

        .. testsetup::

            >>> import torch
            >>> from torchvision import io
            >>> data = torch.randint(255, (10, 64, 64, 3))
            >>> _ = [io.write_video(f"video_{i}.mp4", data, 5, "libx264rgb", {"crf": "0"}) for i in range(1, 4)]
            >>> _ = [io.write_video(f"predict_video_{i}.mp4", data, 5, "libx264rgb", {"crf": "0"}) for i in range(1, 4)]

        .. doctest::

            >>> import fiftyone as fo
            >>> from flash import Trainer
            >>> from flash.video import VideoClassifier, VideoClassificationData
            >>> train_dataset = fo.Dataset.from_videos(
            ...     ["video_1.mp4", "video_2.mp4", "video_3.mp4"]
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            <BLANKLINE>
            ...
            >>> samples = [train_dataset[filepath] for filepath in train_dataset.values("filepath")]
            >>> for sample, label in zip(samples, ["cat", "dog", "cat"]):
            ...     sample["ground_truth"] = fo.Classification(label=label)
            ...     sample.save()
            ...
            >>> predict_dataset = fo.Dataset.from_images(
            ...     ["predict_video_1.mp4", "predict_video_2.mp4", "predict_video_3.mp4"]
            ... )  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            <BLANKLINE>
            ...
            >>> datamodule = VideoClassificationData.from_fiftyone(
            ...     train_dataset=train_dataset,
            ...     predict_dataset=predict_dataset,
            ...     transform_kwargs=dict(image_size=(244, 244)),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = VideoClassifier(backbone="x3d_xs", num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> import os
            >>> _ = [os.remove(f"video_{i}.mp4") for i in range(1, 4)]
            >>> _ = [os.remove(f"predict_video_{i}.mp4") for i in range(1, 4)]
            >>> del train_dataset
            >>> del predict_dataset
        """
        ds_kw = dict(
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            decode_audio=decode_audio,
            decoder=decoder,
        )

        train_input = input_cls(
            RunningStage.TRAINING,
            train_dataset,
            video_sampler=video_sampler,
            label_field=label_field,
            target_formatter=target_formatter,
            **ds_kw,
        )
        target_formatter = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(
                RunningStage.VALIDATING,
                val_dataset,
                video_sampler=video_sampler,
                label_field=label_field,
                target_formatter=target_formatter,
                **ds_kw,
            ),
            input_cls(
                RunningStage.TESTING,
                test_dataset,
                video_sampler=video_sampler,
                label_field=label_field,
                target_formatter=target_formatter,
                **ds_kw,
            ),
            predict_input_cls(RunningStage.PREDICTING, predict_dataset, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
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
        val_split: Optional[float] = None,
        multi_label: Optional[bool] = False,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
        input_cls: Type[Input] = LabelStudioVideoClassificationInput,
        transform: INPUT_TRANSFORM_TYPE = VideoClassificationInputTransform,
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
            train_export_json: path to label studio export file for train set. (overrides export_json if specified)
            val_export_json: path to label studio export file for validation
            test_export_json: path to label studio export file for test
            predict_export_json: path to label studio export file for predict
            data_folder: path to label studio data folder
            train_data_folder: path to label studio data folder for train data set. (overrides data_folder if specified)
            val_data_folder: path to label studio data folder for validation data
            test_data_folder: path to label studio data folder for test data
            predict_data_folder: path to label studio data folder for predict data
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            multi_label: Whether the label are multi encoded.
            clip_sampler: Defines how clips should be sampled from each video.
            clip_duration: Defines how long the sampled clips should be for each video.
            clip_sampler_kwargs: Additional keyword arguments to use when constructing the clip sampler.
            video_sampler: Sampler for the internal video container. This defines the order videos are decoded and,
                    if necessary, the distributed split.
            decode_audio: If True, also decode audio from video.
            decoder: Defines what type of decoder used to decode a video.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
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
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            video_sampler=video_sampler,
            decode_audio=decode_audio,
            decoder=decoder,
        )

        train_input = input_cls(RunningStage.TRAINING, train_data, **ds_kw)
        ds_kw["parameters"] = getattr(train_input, "parameters", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_data, **ds_kw),
            input_cls(RunningStage.TESTING, test_data, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )
