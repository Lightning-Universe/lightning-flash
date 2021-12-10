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
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type, TYPE_CHECKING, Union

import numpy as np
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import Sampler

from flash.core.data.data_module import DataModule
from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.io.classification_input import ClassificationState
from flash.core.data.io.input import DataKeys, Input, InputFormat, IterableInput
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.utilities.paths import list_valid_files
from flash.core.integrations.fiftyone.utils import FiftyOneLabelUtilities
from flash.core.integrations.labelstudio.input import _parse_labelstudio_arguments, LabelStudioVideoClassificationInput
from flash.core.utilities.imports import (
    _FIFTYONE_AVAILABLE,
    _KORNIA_AVAILABLE,
    _PYTORCHVIDEO_AVAILABLE,
    lazy_import,
    requires,
)
from flash.core.utilities.stages import RunningStage

SampleCollection = None
if _FIFTYONE_AVAILABLE:
    fol = lazy_import("fiftyone.core.labels")
    if TYPE_CHECKING:
        from fiftyone.core.collections import SampleCollection
else:
    fol = None

if _KORNIA_AVAILABLE:
    import kornia.augmentation as K

if _PYTORCHVIDEO_AVAILABLE:
    from pytorchvideo.data.clip_sampling import ClipSampler, make_clip_sampler
    from pytorchvideo.data.encoded_video import EncodedVideo
    from pytorchvideo.data.labeled_video_dataset import labeled_video_dataset, LabeledVideoDataset
    from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
    from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample
    from torchvision.transforms import CenterCrop, Compose, RandomCrop, RandomHorizontalFlip
else:
    ClipSampler, LabeledVideoDataset, EncodedVideo, ApplyTransformToKey = None, None, None, None

_PYTORCHVIDEO_DATA = Dict[str, Union[str, torch.Tensor, int, float, List]]

Label = Union[int, List[int]]


def _make_clip_sampler(
    clip_sampler: Union[str, "ClipSampler"] = "random",
    clip_duration: float = 2,
    clip_sampler_kwargs: Dict[str, Any] = None,
) -> "ClipSampler":
    if clip_sampler_kwargs is None:
        clip_sampler_kwargs = {}
    return make_clip_sampler(clip_sampler, clip_duration, **clip_sampler_kwargs)


class VideoClassificationInput(IterableInput):
    def load_data(self, dataset: "LabeledVideoDataset") -> "LabeledVideoDataset":
        if self.training:
            label_to_class_mapping = {p[1]: p[0].split("/")[-2] for p in dataset._labeled_videos._paths_and_labels}
            self.set_state(ClassificationState(label_to_class_mapping))
            self.num_classes = len(np.unique([s[1]["label"] for s in dataset._labeled_videos]))
        return dataset

    def load_sample(self, sample):
        return sample


class VideoClassificationPathsPredictInput(Input):
    def predict_load_data(
        self,
        paths: List[str],
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        decode_audio: bool = False,
        decoder: str = "pyav",
        **_: Any,
    ) -> Iterable[Tuple[str, Any]]:
        paths = list_valid_files(paths, valid_extensions=("mp4", "avi"))
        self._clip_sampler = _make_clip_sampler(clip_sampler, clip_duration, clip_sampler_kwargs)
        self._decode_audio = decode_audio
        self._decoder = decoder
        return paths

    def predict_load_sample(self, sample: str) -> Dict[str, Any]:
        video = EncodedVideo.from_path(sample, decode_audio=self._decode_audio, decoder=self._decoder)
        (
            clip_start,
            clip_end,
            clip_index,
            aug_index,
            is_last_clip,
        ) = self._clip_sampler(0.0, video.duration, None)

        loaded_clip = video.get_clip(clip_start, clip_end)

        clip_is_null = (
            loaded_clip is None or loaded_clip["video"] is None or (loaded_clip["audio"] is None and self._decode_audio)
        )

        if clip_is_null:
            raise MisconfigurationException(
                f"The provided video is too short {video.duration} to be clipped at {self._clip_sampler._clip_duration}"
            )

        frames = loaded_clip["video"]
        audio_samples = loaded_clip["audio"]
        return {
            "video": frames,
            "video_name": video.name,
            "video_index": 0,
            "clip_index": clip_index,
            "aug_index": aug_index,
            **({"audio": audio_samples} if audio_samples is not None else {}),
            DataKeys.METADATA: {"filepath": sample},
        }


class VideoClassificationFoldersInput(VideoClassificationInput):
    def load_data(
        self,
        path: str,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
    ) -> "LabeledVideoDataset":
        dataset = labeled_video_dataset(
            pathlib.Path(path),
            _make_clip_sampler(clip_sampler, clip_duration, clip_sampler_kwargs),
            video_sampler=video_sampler,
            decode_audio=decode_audio,
            decoder=decoder,
        )
        return super().load_data(dataset)


class VideoClassificationFilesInput(VideoClassificationInput):
    def _to_multi_hot(self, label_list: List[int]) -> torch.Tensor:
        v = torch.zeros(len(self.labels_set))
        for label in label_list:
            v[label] = 1
        return v

    def load_data(
        self,
        paths: List[str],
        labels: List[Union[str, List]],
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
    ) -> "LabeledVideoDataset":
        self.is_multilabel = any(isinstance(label, list) for label in labels)
        if self.is_multilabel:
            self.labels_set = {label for label_list in labels for label in label_list}
            self.label_to_id = {label: i for i, label in enumerate(sorted(self.labels_set))}
            self.id_to_label = {i: label for label, i in self.label_to_id.items()}

            encoded_labels = [
                self._to_multi_hot([self.label_to_id[classname] for classname in label_list]) for label_list in labels
            ]

            data = list(
                zip(
                    paths,
                    encoded_labels,
                )
            )
        else:
            self.labels_set = set(labels)
            self.label_to_id = {label: i for i, label in enumerate(sorted(self.labels_set))}
            self.id_to_label = {i: label for label, i in self.label_to_id.items()}
            data = list(zip(paths, [self.label_to_id[classname] for classname in labels]))

        dataset = LabeledVideoDataset(
            LabeledVideoPaths(data),
            _make_clip_sampler(clip_sampler, clip_duration, clip_sampler_kwargs),
            video_sampler=video_sampler,
            decode_audio=decode_audio,
            decoder=decoder,
        )
        if self.training:
            self.set_state(ClassificationState(self.id_to_label))
            self.num_classes = len(self.labels_set)
        return dataset


class VideoClassificationFiftyOneInput(VideoClassificationInput):
    def load_data(
        self,
        sample_collection: SampleCollection,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
        label_field: str = "ground_truth",
    ) -> "LabeledVideoDataset":
        label_utilities = FiftyOneLabelUtilities(label_field, fol.Classification)
        label_utilities.validate(sample_collection)
        classes = label_utilities.get_classes(sample_collection)
        label_to_class_mapping = dict(enumerate(classes))
        class_to_label_mapping = {c: lab for lab, c in label_to_class_mapping.items()}

        filepaths = sample_collection.values("filepath")
        labels = sample_collection.values(label_field + ".label")
        targets = [class_to_label_mapping[lab] for lab in labels]

        dataset = LabeledVideoDataset(
            LabeledVideoPaths(list(zip(filepaths, targets))),
            _make_clip_sampler(clip_sampler, clip_duration, clip_sampler_kwargs),
            video_sampler=video_sampler,
            decode_audio=decode_audio,
            decoder=decoder,
        )
        return super().load_data(dataset)


class VideoClassificationInputTransform(InputTransform):
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
        **_kwargs: Any,
    ):
        self.clip_sampler = clip_sampler
        self.clip_duration = clip_duration
        self.clip_sampler_kwargs = clip_sampler_kwargs
        self.video_sampler = video_sampler
        self.decode_audio = decode_audio
        self.decoder = decoder

        if not _PYTORCHVIDEO_AVAILABLE:
            raise ModuleNotFoundError("Please, run `pip install pytorchvideo`.")

        if not clip_sampler_kwargs:
            clip_sampler_kwargs = {}

        if not clip_sampler:
            raise MisconfigurationException(
                "clip_sampler should be provided as a string or ``pytorchvideo.data.clip_sampling.ClipSampler``"
            )

        clip_sampler = make_clip_sampler(clip_sampler, clip_duration, **clip_sampler_kwargs)

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            inputs={
                InputFormat.FILES: VideoClassificationPathsPredictInput,
                InputFormat.FOLDERS: VideoClassificationPathsPredictInput,
                InputFormat.FIFTYONE: VideoClassificationFiftyOneInput,
            },
            default_input=InputFormat.FILES,
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            **self.transforms,
            "clip_sampler": self.clip_sampler,
            "clip_duration": self.clip_duration,
            "clip_sampler_kwargs": self.clip_sampler_kwargs,
            "video_sampler": self.video_sampler,
            "decode_audio": self.decode_audio,
            "decoder": self.decoder,
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool) -> "VideoClassificationInputTransform":
        return cls(**state_dict)

    def default_transforms(self) -> Dict[str, Callable]:
        if self.training:
            per_sample_transform = [
                RandomCrop(244, pad_if_needed=True),
                RandomHorizontalFlip(p=0.5),
            ]
        else:
            per_sample_transform = [
                CenterCrop(244),
            ]

        return {
            "per_sample_transform": Compose(
                [
                    ApplyTransformToKey(
                        key="video",
                        transform=Compose([UniformTemporalSubsample(8)] + per_sample_transform),
                    ),
                ]
            ),
            "per_batch_transform_on_device": Compose(
                [
                    ApplyTransformToKey(
                        key="video",
                        transform=K.VideoSequential(
                            K.Normalize(torch.tensor([0.45, 0.45, 0.45]), torch.tensor([0.225, 0.225, 0.225])),
                            data_format="BCTHW",
                            same_on_frame=False,
                        ),
                    ),
                ]
            ),
        }


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
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
        **data_module_kwargs,
    ) -> "VideoClassificationData":
        dataset_kwargs = dict(
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            video_sampler=video_sampler,
            decode_audio=decode_audio,
            decoder=decoder,
            data_pipeline_state=DataPipelineState(),
        )
        return cls(
            VideoClassificationFilesInput(RunningStage.TRAINING, train_files, train_targets, **dataset_kwargs),
            VideoClassificationFilesInput(RunningStage.VALIDATING, val_files, val_targets, **dataset_kwargs),
            VideoClassificationFilesInput(RunningStage.TESTING, test_files, test_targets, **dataset_kwargs),
            VideoClassificationPathsPredictInput(RunningStage.PREDICTING, predict_files, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
            ),
            **data_module_kwargs,
        )

    @classmethod
    def from_folders(
        cls,
        train_folder: Optional[str] = None,
        val_folder: Optional[str] = None,
        test_folder: Optional[str] = None,
        predict_folder: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
        **data_module_kwargs,
    ) -> "VideoClassificationData":
        dataset_kwargs = dict(
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            video_sampler=video_sampler,
            decode_audio=decode_audio,
            decoder=decoder,
            data_pipeline_state=DataPipelineState(),
        )
        return cls(
            VideoClassificationFoldersInput(RunningStage.TRAINING, train_folder, **dataset_kwargs),
            VideoClassificationFoldersInput(RunningStage.VALIDATING, val_folder, **dataset_kwargs),
            VideoClassificationFoldersInput(RunningStage.TESTING, test_folder, **dataset_kwargs),
            VideoClassificationPathsPredictInput(RunningStage.PREDICTING, predict_folder, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
            ),
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
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
        label_field: str = "ground_truth",
        **data_module_kwargs,
    ) -> "VideoClassificationData":
        dataset_kwargs = dict(
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            video_sampler=video_sampler,
            decode_audio=decode_audio,
            decoder=decoder,
            label_field=label_field,
            data_pipeline_state=DataPipelineState(),
        )
        return cls(
            VideoClassificationFiftyOneInput(RunningStage.TRAINING, train_dataset, **dataset_kwargs),
            VideoClassificationFiftyOneInput(RunningStage.VALIDATING, val_dataset, **dataset_kwargs),
            VideoClassificationFiftyOneInput(RunningStage.TESTING, test_dataset, **dataset_kwargs),
            VideoClassificationFiftyOneInput(RunningStage.PREDICTING, predict_dataset, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
            ),
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
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        val_split: Optional[float] = None,
        multi_label: Optional[bool] = False,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
        **data_module_kwargs: Any,
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

        dataset_kwargs = dict(
            data_pipeline_state=DataPipelineState(),
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            video_sampler=video_sampler,
            decode_audio=decode_audio,
            decoder=decoder,
        )

        return cls(
            LabelStudioVideoClassificationInput(RunningStage.TRAINING, train_data, **dataset_kwargs),
            LabelStudioVideoClassificationInput(RunningStage.VALIDATING, val_data, **dataset_kwargs),
            LabelStudioVideoClassificationInput(RunningStage.TESTING, test_data, **dataset_kwargs),
            LabelStudioVideoClassificationInput(RunningStage.PREDICTING, predict_data, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform,
                val_transform,
                test_transform,
                predict_transform,
                **data_module_kwargs,
            ),
            **data_module_kwargs,
        )
