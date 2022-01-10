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
from typing import Any, Dict, Iterable, List, Tuple, Type, Union

import numpy as np
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import Sampler

from flash.core.data.io.classification_input import ClassificationState
from flash.core.data.io.input import DataKeys, Input, IterableInput
from flash.core.data.utilities.paths import list_valid_files
from flash.core.integrations.fiftyone.utils import FiftyOneLabelUtilities
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, _PYTORCHVIDEO_AVAILABLE, lazy_import

if _FIFTYONE_AVAILABLE:
    fol = lazy_import("fiftyone.core.labels")
    SampleCollection = lazy_import("fiftyone.core.collections.SampleCollection")
else:
    fol = None
    SampleCollection = None

if _PYTORCHVIDEO_AVAILABLE:
    from pytorchvideo.data.clip_sampling import ClipSampler, make_clip_sampler
    from pytorchvideo.data.encoded_video import EncodedVideo
    from pytorchvideo.data.labeled_video_dataset import labeled_video_dataset, LabeledVideoDataset
    from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
else:
    ClipSampler, LabeledVideoDataset, EncodedVideo, ApplyTransformToKey = None, None, None, None


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
