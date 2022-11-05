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
import os
from typing import Any, Callable, Collection, Dict, List, Optional, Type, Union

import pandas as pd
import torch
from torch.utils.data import Sampler

from flash.core.data.io.classification_input import ClassificationInputMixin
from flash.core.data.io.input import DataKeys, Input, IterableInput
from flash.core.data.utilities.classification import _is_list_like, MultiBinaryTargetFormatter, TargetFormatter
from flash.core.data.utilities.data_frame import resolve_files, resolve_targets
from flash.core.data.utilities.loading import load_data_frame
from flash.core.data.utilities.paths import list_valid_files, make_dataset, PATH_TYPE
from flash.core.integrations.fiftyone.utils import FiftyOneLabelUtilities
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, _PYTORCHVIDEO_AVAILABLE, lazy_import, requires

if _FIFTYONE_AVAILABLE:
    fol = lazy_import("fiftyone.core.labels")
    SampleCollection = "fiftyone.core.collections.SampleCollection"
else:
    fol = None
    SampleCollection = None

if _PYTORCHVIDEO_AVAILABLE:
    from pytorchvideo.data.clip_sampling import ClipSampler, make_clip_sampler
    from pytorchvideo.data.encoded_video import EncodedVideo
    from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset
    from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths

    from flash.video.classification.utils import LabeledVideoTensorDataset

else:
    ClipSampler, LabeledVideoDataset, LabeledVideoTensorDataset, EncodedVideo, ApplyTransformToKey = (
        None,
        None,
        None,
        None,
        None,
    )


def _make_clip_sampler(
    clip_sampler: Union[str, "ClipSampler"] = "random",
    clip_duration: float = 2,
    clip_sampler_kwargs: Dict[str, Any] = None,
) -> "ClipSampler":
    if clip_sampler_kwargs is None:
        clip_sampler_kwargs = {}
    return make_clip_sampler(clip_sampler, clip_duration, **clip_sampler_kwargs)


class VideoClassificationInput(IterableInput, ClassificationInputMixin):
    def load_data(
        self,
        files: List[PATH_TYPE],
        targets: List[Any],
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
        target_formatter: Optional[TargetFormatter] = None,
    ) -> "LabeledVideoDataset":
        dataset = LabeledVideoDataset(
            LabeledVideoPaths(list(zip(files, targets))),
            _make_clip_sampler(clip_sampler, clip_duration, clip_sampler_kwargs),
            video_sampler=video_sampler,
            decode_audio=decode_audio,
            decoder=decoder,
        )
        if not self.predicting:
            self.load_target_metadata(
                [sample[1] for sample in dataset._labeled_videos._paths_and_labels], target_formatter=target_formatter
            )
        return dataset

    def load_sample(self, sample):
        sample["label"] = self.format_target(sample["label"])
        sample[DataKeys.INPUT] = sample.pop("video")
        sample[DataKeys.TARGET] = sample.pop("label")
        return sample


class VideoClassificationTensorsBaseInput(IterableInput, ClassificationInputMixin):
    def load_data(
        self,
        inputs: Optional[Union[Collection[torch.Tensor], torch.Tensor]],
        targets: Union[List[Any], Any],
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        target_formatter: Optional[TargetFormatter] = None,
    ) -> "LabeledVideoTensorDataset":
        if isinstance(inputs, torch.Tensor):
            # In case of (number of videos x CTHW) format
            if inputs.ndim == 5:
                inputs = list(inputs)
            elif inputs.ndim == 4:
                inputs = [inputs]
            else:
                raise ValueError(
                    f"Got dimension of the input tensor: {inputs.ndim}"
                    " for stack of tensors - dimension should be 5 or for a single tensor, dimension should be 4.",
                )
        elif not _is_list_like(inputs):
            raise TypeError(f"Expected either a list/tuple of torch.Tensor or torch.Tensor, but got: {type(inputs)}.")

        # Note: We take whatever is the shortest out of inputs and targets
        dataset = LabeledVideoTensorDataset(list(zip(inputs, targets)), video_sampler=video_sampler)
        if not self.predicting:
            self.load_target_metadata(
                [sample[1] for sample in dataset._labeled_videos], target_formatter=target_formatter
            )
        return dataset

    def load_sample(self, sample):
        sample["label"] = self.format_target(sample["label"])
        sample[DataKeys.INPUT] = sample.pop("video")
        sample[DataKeys.TARGET] = sample.pop("label")
        return sample


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
        target_formatter: Optional[TargetFormatter] = None,
    ) -> "LabeledVideoDataset":
        return super().load_data(
            *make_dataset(path, extensions=("mp4", "avi")),
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            video_sampler=video_sampler,
            decode_audio=decode_audio,
            decoder=decoder,
            target_formatter=target_formatter,
        )


class VideoClassificationFilesInput(VideoClassificationInput):
    def load_data(
        self,
        paths: List[str],
        targets: List[Any],
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
        target_formatter: Optional[TargetFormatter] = None,
    ) -> "LabeledVideoDataset":
        return super().load_data(
            paths,
            targets,
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            video_sampler=video_sampler,
            decode_audio=decode_audio,
            decoder=decoder,
            target_formatter=target_formatter,
        )


class VideoClassificationDataFrameInput(VideoClassificationInput):
    labels: list

    def load_data(
        self,
        data_frame: pd.DataFrame,
        input_key: str,
        target_keys: Union[str, List[str]],
        root: Optional[PATH_TYPE] = None,
        resolver: Optional[Callable[[Optional[PATH_TYPE], Any], PATH_TYPE]] = None,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
        target_formatter: Optional[TargetFormatter] = None,
    ) -> "LabeledVideoDataset":
        result = super().load_data(
            resolve_files(data_frame, input_key, root, resolver),
            resolve_targets(data_frame, target_keys),
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            video_sampler=video_sampler,
            decode_audio=decode_audio,
            decoder=decoder,
            target_formatter=target_formatter,
        )

        # If we had binary multi-class targets then we also know the labels (column names)
        if (
            self.training
            and isinstance(self.target_formatter, MultiBinaryTargetFormatter)
            and isinstance(target_keys, List)
        ):
            self.labels = target_keys

        return result


class VideoClassificationTensorsInput(VideoClassificationTensorsBaseInput):
    labels: list

    def load_data(
        self,
        tensors: Any,
        targets: Optional[List[Any]] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        target_formatter: Optional[TargetFormatter] = None,
    ) -> "LabeledVideoTensorDataset":
        result = super().load_data(
            tensors,
            targets,
            video_sampler=video_sampler,
            target_formatter=target_formatter,
        )

        # If we had binary multi-class targets then we also know the labels (column names)
        if (
            self.training
            and isinstance(self.target_formatter, MultiBinaryTargetFormatter)
            and isinstance(targets, List)
        ):
            self.labels = targets

        return result


class VideoClassificationCSVInput(VideoClassificationDataFrameInput):
    def load_data(
        self,
        csv_file: PATH_TYPE,
        input_key: str,
        target_keys: Optional[Union[str, List[str]]] = None,
        root: Optional[PATH_TYPE] = None,
        resolver: Optional[Callable[[Optional[PATH_TYPE], Any], PATH_TYPE]] = None,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        video_sampler: Type[Sampler] = torch.utils.data.RandomSampler,
        decode_audio: bool = False,
        decoder: str = "pyav",
        target_formatter: Optional[TargetFormatter] = None,
    ) -> "LabeledVideoDataset":
        data_frame = load_data_frame(csv_file)
        if root is None:
            root = os.path.dirname(csv_file)
        return super().load_data(
            data_frame,
            input_key,
            target_keys,
            root,
            resolver,
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            video_sampler=video_sampler,
            decode_audio=decode_audio,
            decoder=decoder,
            target_formatter=target_formatter,
        )


class VideoClassificationFiftyOneInput(VideoClassificationInput):
    @requires("fiftyone")
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
        target_formatter: Optional[TargetFormatter] = None,
    ) -> "LabeledVideoDataset":
        label_utilities = FiftyOneLabelUtilities(label_field, fol.Classification)
        label_utilities.validate(sample_collection)

        return super().load_data(
            sample_collection.values("filepath"),
            sample_collection.values(label_field + ".label"),
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            video_sampler=video_sampler,
            decode_audio=decode_audio,
            decoder=decoder,
            target_formatter=target_formatter,
        )


class VideoClassificationPathsPredictInput(Input):
    def predict_load_data(
        self,
        paths: List[str],
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        decode_audio: bool = False,
        decoder: str = "pyav",
    ) -> List[str]:
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
            raise ValueError(
                f"The provided video is too short {video.duration} to be clipped at {self._clip_sampler._clip_duration}"
            )

        frames = loaded_clip["video"]
        audio_samples = loaded_clip["audio"]
        return {
            DataKeys.INPUT: frames,
            "video_name": video.name,
            "video_index": 0,
            "clip_index": clip_index,
            "aug_index": aug_index,
            **({"audio": audio_samples} if audio_samples is not None else {}),
            DataKeys.METADATA: {"filepath": sample},
        }


class VideoClassificationDataFramePredictInput(VideoClassificationPathsPredictInput):
    def predict_load_data(
        self,
        data_frame: pd.DataFrame,
        input_key: str,
        root: Optional[PATH_TYPE] = None,
        resolver: Optional[Callable[[Optional[PATH_TYPE], Any], PATH_TYPE]] = None,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        decode_audio: bool = False,
        decoder: str = "pyav",
    ) -> List[str]:
        return super().predict_load_data(
            resolve_files(data_frame, input_key, root, resolver),
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            decode_audio=decode_audio,
            decoder=decoder,
        )


class VideoClassificationTensorsPredictInput(Input):
    def predict_load_data(self, data: Union[torch.Tensor, List[Any], Any]):
        if _is_list_like(data):
            return data
        else:
            if not isinstance(data, torch.Tensor):
                raise TypeError(f"Expected either a list/tuple of torch.Tensor or torch.Tensor, but got: {type(data)}.")
            if data.ndim == 5:
                return list(data)
            elif data.ndim == 4:
                return [data]
            else:
                raise ValueError(
                    f"Got dimension of the input tensor: {data.ndim},"
                    " for stack of tensors - dimension should be 5 or for a single tensor, dimension should be 4."
                )

    def predict_load_sample(self, sample: torch.Tensor) -> Dict[str, Any]:
        return {
            DataKeys.INPUT: sample,
            "video_index": 0,
        }


class VideoClassificationCSVPredictInput(VideoClassificationDataFramePredictInput):
    def predict_load_data(
        self,
        csv_file: PATH_TYPE,
        input_key: str,
        root: Optional[PATH_TYPE] = None,
        resolver: Optional[Callable[[Optional[PATH_TYPE], Any], PATH_TYPE]] = None,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        decode_audio: bool = False,
        decoder: str = "pyav",
    ) -> List[str]:
        data_frame = load_data_frame(csv_file)
        if root is None:
            root = os.path.dirname(csv_file)
        return super().predict_load_data(
            data_frame,
            input_key,
            root,
            resolver,
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            decode_audio=decode_audio,
            decoder=decoder,
        )


class VideoClassificationFiftyOnePredictInput(VideoClassificationPathsPredictInput):
    @requires("fiftyone")
    def predict_load_data(
        self,
        data: SampleCollection,
        clip_sampler: Union[str, "ClipSampler"] = "random",
        clip_duration: float = 2,
        clip_sampler_kwargs: Dict[str, Any] = None,
        decode_audio: bool = False,
        decoder: str = "pyav",
    ) -> List[str]:
        return super().predict_load_data(
            data.values("filepath"),
            clip_sampler=clip_sampler,
            clip_duration=clip_duration,
            clip_sampler_kwargs=clip_sampler_kwargs,
            decode_audio=decode_audio,
            decoder=decoder,
        )
