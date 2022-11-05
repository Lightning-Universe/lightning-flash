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
from flash.core.data.utilities.loading import AUDIO_EXTENSIONS, load_audio, load_data_frame
from flash.core.data.utilities.paths import filter_valid_files, list_valid_files
from flash.core.data.utilities.samples import to_sample, to_samples
from flash.core.utilities.imports import _AUDIO_AVAILABLE, requires

if _AUDIO_AVAILABLE:
    import librosa
    from datasets import Dataset as HFDataset
    from datasets import load_dataset
else:
    HFDataset = object


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


class SpeechRecognitionInputBase(Input):
    sampling_rate: int

    @requires("audio")
    def load_data(
        self,
        hf_dataset: HFDataset,
        root: str,
        input_key: str,
        target_key: Optional[str] = None,
        sampling_rate: int = 16000,
        filetype: Optional[str] = None,
    ) -> Sequence[Mapping[str, Any]]:
        self.sampling_rate = sampling_rate

        meta = {"root": root}
        if target_key is not None:
            return [
                {
                    DataKeys.INPUT: input_file,
                    DataKeys.TARGET: target,
                    DataKeys.METADATA: meta,
                }
                for input_file, target in zip(hf_dataset[input_key], hf_dataset[target_key])
            ]
        return [
            {
                DataKeys.INPUT: input_file,
                DataKeys.METADATA: meta,
            }
            for input_file in hf_dataset[input_key]
        ]

    def load_sample(self, sample: Dict[str, Any]) -> Any:
        path = sample[DataKeys.INPUT]
        if not os.path.isabs(path) and DataKeys.METADATA in sample and "root" in sample[DataKeys.METADATA]:
            path = os.path.join(sample[DataKeys.METADATA]["root"], path)
        speech_array = load_audio(path, sampling_rate=self.sampling_rate)
        sample[DataKeys.INPUT] = speech_array
        sample[DataKeys.METADATA] = {"sampling_rate": self.sampling_rate}
        return sample


class SpeechRecognitionCSVInput(SpeechRecognitionInputBase):
    @requires("audio")
    def load_data(
        self,
        csv_file: str,
        input_key: str,
        target_key: Optional[str] = None,
        sampling_rate: int = 16000,
    ):
        return super().load_data(
            HFDataset.from_pandas(load_data_frame(csv_file)),
            os.path.dirname(csv_file),
            input_key,
            target_key,
            sampling_rate=sampling_rate,
        )


class SpeechRecognitionJSONInput(SpeechRecognitionInputBase):
    @requires("audio")
    def load_data(
        self,
        json_file: str,
        input_key: str,
        target_key: Optional[str] = None,
        field: Optional[str] = None,
        sampling_rate: int = 16000,
    ):
        dataset_dict = load_dataset("json", data_files={"data": str(json_file)}, field=field)
        return super().load_data(
            dataset_dict["data"],
            os.path.dirname(json_file),
            input_key,
            target_key,
            sampling_rate=sampling_rate,
            filetype="json",
        )


class SpeechRecognitionDatasetInput(SpeechRecognitionInputBase):
    sampling_rate: int

    @requires("audio")
    def load_data(self, dataset: Dataset, sampling_rate: int = 16000) -> Sequence[Mapping[str, Any]]:
        self.sampling_rate = sampling_rate
        return dataset

    def load_sample(self, sample: Any) -> Any:
        sample = to_sample(sample)
        if isinstance(sample[DataKeys.INPUT], (str, Path)):
            sample = super().load_sample(sample)
        return sample


class SpeechRecognitionPathsInput(SpeechRecognitionInputBase):
    @requires("audio")
    def load_data(
        self,
        paths: Union[str, List[str]],
        targets: Optional[List[str]] = None,
        sampling_rate: int = 16000,
    ) -> Sequence:
        self.sampling_rate = sampling_rate
        if targets is None:
            return to_samples(list_valid_files(paths, AUDIO_EXTENSIONS))
        return to_samples(*filter_valid_files(paths, targets, valid_extensions=AUDIO_EXTENSIONS))
