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
import base64
import io
import os.path
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from torch.utils.data import Dataset

import flash
from flash.core.data.io.input import DataKeys, Input, ServeInput
from flash.core.data.utilities.paths import filter_valid_files, list_valid_files
from flash.core.data.utilities.samples import to_sample, to_samples
from flash.core.utilities.imports import _AUDIO_AVAILABLE, requires

if _AUDIO_AVAILABLE:
    import librosa
    from datasets import load_dataset


class SpeechRecognitionDeserializer(ServeInput):
    @requires("audio")
    def __init__(self, sampling_rate: int = 16000, **kwargs):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate

    def serve_load_sample(self, sample: Any) -> Dict:
        encoded_with_padding = (sample + "===").encode("ascii")
        audio = base64.b64decode(encoded_with_padding)
        buffer = io.BytesIO(audio)
        data, sampling_rate = librosa.load(buffer, sr=self.sampling_rate)
        return {
            DataKeys.INPUT: data,
            DataKeys.METADATA: {"sampling_rate": sampling_rate},
        }

    @property
    def example_input(self) -> str:
        with (Path(flash.ASSETS_ROOT) / "example.wav").open("rb") as f:
            return base64.b64encode(f.read()).decode("UTF-8")


class BaseSpeechRecognition(Input):
    @staticmethod
    def load_sample(sample: Dict[str, Any], sampling_rate: int = 16000) -> Any:
        path = sample[DataKeys.INPUT]
        if not os.path.isabs(path) and DataKeys.METADATA in sample and "root" in sample[DataKeys.METADATA]:
            path = os.path.join(sample[DataKeys.METADATA]["root"], path)
        speech_array, sampling_rate = librosa.load(path, sr=sampling_rate)
        sample[DataKeys.INPUT] = speech_array
        sample[DataKeys.METADATA] = {"sampling_rate": sampling_rate}
        return sample


class SpeechRecognitionFileInput(BaseSpeechRecognition):
    @requires("audio")
    def load_data(
        self,
        file: str,
        input_key: str,
        target_key: Optional[str] = None,
        field: Optional[str] = None,
        sampling_rate: int = 16000,
        filetype: Optional[str] = None,
    ) -> Sequence[Mapping[str, Any]]:
        self.sampling_rate = sampling_rate

        stage = self.running_stage.value
        if filetype == "json" and field is not None:
            dataset_dict = load_dataset(filetype, data_files={stage: str(file)}, field=field)
        else:
            dataset_dict = load_dataset(filetype, data_files={stage: str(file)})

        dataset = dataset_dict[stage]
        meta = {"root": os.path.dirname(file)}
        if target_key is not None:
            return [
                {
                    DataKeys.INPUT: input_file,
                    DataKeys.TARGET: target,
                    DataKeys.METADATA: meta,
                }
                for input_file, target in zip(dataset[input_key], dataset[target_key])
            ]
        return [
            {
                DataKeys.INPUT: input_file,
                DataKeys.METADATA: meta,
            }
            for input_file in dataset[input_key]
        ]

    def load_sample(self, sample: Dict[str, Any]) -> Any:
        return super().load_sample(sample, self.sampling_rate)


class SpeechRecognitionCSVInput(SpeechRecognitionFileInput):
    @requires("audio")
    def load_data(
        self,
        file: str,
        input_key: str,
        target_key: Optional[str] = None,
        sampling_rate: int = 16000,
    ):
        return super().load_data(file, input_key, target_key, sampling_rate=sampling_rate, filetype="csv")


class SpeechRecognitionJSONInput(SpeechRecognitionFileInput):
    @requires("audio")
    def load_data(
        self,
        file: str,
        input_key: str,
        target_key: Optional[str] = None,
        field: Optional[str] = None,
        sampling_rate: int = 16000,
    ):
        return super().load_data(file, input_key, target_key, field, sampling_rate=sampling_rate, filetype="json")


class SpeechRecognitionDatasetInput(BaseSpeechRecognition):
    @requires("audio")
    def load_data(self, dataset: Dataset, sampling_rate: int = 16000) -> Sequence[Mapping[str, Any]]:
        self.sampling_rate = sampling_rate
        return super().load_data(dataset)

    def load_sample(self, sample: Any) -> Any:
        sample = to_sample(sample)
        if isinstance(sample[DataKeys.INPUT], (str, Path)):
            sample = super().load_sample(sample, self.sampling_rate)
        return sample


class SpeechRecognitionPathsInput(BaseSpeechRecognition):
    @requires("audio")
    def load_data(
        self,
        paths: Union[str, List[str]],
        targets: Optional[List[str]] = None,
        sampling_rate: int = 16000,
    ) -> Sequence:
        self.sampling_rate = sampling_rate
        if targets is None:
            return to_samples(list_valid_files(paths, ("wav", "ogg", "flac", "mat", "mp3")))
        return to_samples(*filter_valid_files(paths, targets, valid_extensions=("wav", "ogg", "flac", "mat", "mp3")))

    def load_sample(self, sample: Dict[str, Any]) -> Any:
        return super().load_sample(sample, self.sampling_rate)
