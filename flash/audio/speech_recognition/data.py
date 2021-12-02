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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

import flash
from flash.core.data.data_module import DataModule
from flash.core.data.io.input import DataKeys, InputFormat
from flash.core.data.io.input_base import Input
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.io.output_transform import OutputTransform
from flash.core.data.process import Deserializer
from flash.core.data.properties import ProcessState
from flash.core.data.utilities.paths import list_valid_files
from flash.core.utilities.imports import _AUDIO_AVAILABLE, requires
from flash.core.utilities.stages import RunningStage

if _AUDIO_AVAILABLE:
    import librosa
    from datasets import Dataset as HFDataset
    from datasets import load_dataset
    from transformers import Wav2Vec2CTCTokenizer
else:
    HFDataset = object


class SpeechRecognitionDeserializer(Deserializer):
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
        target_key: str,
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
        return [
            {
                DataKeys.INPUT: input_file,
                DataKeys.TARGET: target,
                DataKeys.METADATA: meta,
            }
            for input_file, target in zip(dataset[input_key], dataset[target_key])
        ]

    def load_sample(self, sample: Dict[str, Any]) -> Any:
        return super().load_sample(sample, self.sampling_rate)


class SpeechRecognitionCSVInput(SpeechRecognitionFileInput):
    @requires("audio")
    def load_data(
        self,
        file: str,
        input_key: str,
        target_key: str,
        sampling_rate: int = 16000,
    ):
        return super().load_data(file, input_key, target_key, sampling_rate=sampling_rate, filetype="csv")


class SpeechRecognitionJSONInput(SpeechRecognitionFileInput):
    @requires("audio")
    def load_data(
        self,
        file: str,
        input_key: str,
        target_key: str,
        field: Optional[str] = None,
        sampling_rate: int = 16000,
    ):
        return super().load_data(file, input_key, target_key, field, sampling_rate=sampling_rate, filetype="json")


class SpeechRecognitionDatasetInput(BaseSpeechRecognition):
    @requires("audio")
    def load_data(self, dataset: Dataset, sampling_rate: int = 16000) -> Sequence[Mapping[str, Any]]:
        self.sampling_rate = sampling_rate
        if isinstance(dataset, HFDataset):
            dataset = list(zip(dataset["file"], dataset["text"]))
        return super().load_data(dataset)

    def load_sample(self, sample: Any) -> Any:
        if isinstance(sample[DataKeys.INPUT], (str, Path)):
            sample = super().load_sample(sample, self.sampling_rate)
        return sample


class SpeechRecognitionPathsInput(BaseSpeechRecognition):
    @requires("audio")
    def load_data(self, paths: Union[str, List[str]], sampling_rate: int = 16000) -> Sequence:
        self.sampling_rate = sampling_rate
        return [{DataKeys.INPUT: file} for file in list_valid_files(paths, ("wav", "ogg", "flac", "mat", "mp3"))]

    def load_sample(self, sample: Dict[str, Any]) -> Any:
        return super().load_sample(sample, self.sampling_rate)


class SpeechRecognitionInputTransform(InputTransform):
    @requires("audio")
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        sampling_rate: int = 16000,
    ):
        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            inputs={
                InputFormat.CSV: SpeechRecognitionCSVInput,
                InputFormat.JSON: SpeechRecognitionJSONInput,
                InputFormat.FILES: SpeechRecognitionPathsInput,
                InputFormat.DATASETS: SpeechRecognitionDatasetInput,
            },
            default_input=InputFormat.FILES,
            deserializer=SpeechRecognitionDeserializer(sampling_rate),
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return self.transforms

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(**state_dict)


@dataclass(unsafe_hash=True, frozen=True)
class SpeechRecognitionBackboneState(ProcessState):
    """The ``SpeechRecognitionBackboneState`` stores the backbone in use by the
    :class:`~flash.audio.speech_recognition.data.SpeechRecognitionOutputTransform`
    """

    backbone: str


class SpeechRecognitionOutputTransform(OutputTransform):
    @requires("audio")
    def __init__(self):
        super().__init__()

        self._backbone = None
        self._tokenizer = None

    @property
    def backbone(self):
        backbone_state = self.get_state(SpeechRecognitionBackboneState)
        if backbone_state is not None:
            return backbone_state.backbone

    @property
    def tokenizer(self):
        if self.backbone is not None and self.backbone != self._backbone:
            self._tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.backbone)
            self._backbone = self.backbone
        return self._tokenizer

    def per_batch_transform(self, batch: Any) -> Any:
        # converts logits into greedy transcription
        pred_ids = torch.argmax(batch.logits, dim=-1)
        transcriptions = self.tokenizer.batch_decode(pred_ids)
        return transcriptions

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("_tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.backbone)


class SpeechRecognitionData(DataModule):
    """Data Module for text classification tasks."""

    input_transform_cls = SpeechRecognitionInputTransform
    output_transform_cls = SpeechRecognitionOutputTransform

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
        sampling_rate: int = 16000,
        **data_module_kwargs,
    ) -> "SpeechRecognitionData":
        return cls(
            SpeechRecognitionPathsInput(RunningStage.TRAINING, train_files, train_targets, sampling_rate=sampling_rate),
            SpeechRecognitionPathsInput(RunningStage.VALIDATING, val_files, val_targets, sampling_rate=sampling_rate),
            SpeechRecognitionPathsInput(RunningStage.TESTING, test_files, test_targets, sampling_rate=sampling_rate),
            SpeechRecognitionPathsInput(RunningStage.PREDICTING, predict_files, sampling_rate=sampling_rate),
            input_transform=cls.input_transform_cls(
                train_transform, val_transform, test_transform, predict_transform, sampling_rate
            ),
            **data_module_kwargs,
        )

    @classmethod
    def from_csv(
        cls,
        input_fields: Union[str, Sequence[str]],
        target_fields: Optional[str] = None,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        sampling_rate: int = 16000,
        **data_module_kwargs: Any,
    ) -> "SpeechRecognitionData":
        dataset_kwargs = dict(
            input_key=input_fields,
            target_key=target_fields,
            sampling_rate=sampling_rate,
        )
        return cls(
            SpeechRecognitionCSVInput(RunningStage.TRAINING, train_file, **dataset_kwargs),
            SpeechRecognitionCSVInput(RunningStage.VALIDATING, val_file, **dataset_kwargs),
            SpeechRecognitionCSVInput(RunningStage.TESTING, test_file, **dataset_kwargs),
            SpeechRecognitionCSVInput(RunningStage.PREDICTING, predict_file, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform, val_transform, test_transform, predict_transform, sampling_rate
            ),
            **data_module_kwargs,
        )

    @classmethod
    def from_json(
        cls,
        input_fields: Union[str, Sequence[str]],
        target_fields: Optional[str] = None,
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        predict_file: Optional[str] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        field: Optional[str] = None,
        sampling_rate: int = 16000,
        **data_module_kwargs: Any,
    ) -> "SpeechRecognitionData":
        dataset_kwargs = dict(
            input_key=input_fields,
            target_key=target_fields,
            sampling_rate=sampling_rate,
            field=field,
        )
        return cls(
            SpeechRecognitionJSONInput(RunningStage.TRAINING, train_file, **dataset_kwargs),
            SpeechRecognitionJSONInput(RunningStage.VALIDATING, val_file, **dataset_kwargs),
            SpeechRecognitionJSONInput(RunningStage.TESTING, test_file, **dataset_kwargs),
            SpeechRecognitionJSONInput(RunningStage.PREDICTING, predict_file, **dataset_kwargs),
            input_transform=cls.input_transform_cls(
                train_transform, val_transform, test_transform, predict_transform, sampling_rate
            ),
            **data_module_kwargs,
        )

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        sampling_rate: int = 16000,
        **data_module_kwargs,
    ) -> "SpeechRecognitionData":
        return cls(
            SpeechRecognitionDatasetInput(RunningStage.TRAINING, train_dataset, sampling_rate=sampling_rate),
            SpeechRecognitionDatasetInput(RunningStage.VALIDATING, val_dataset, sampling_rate=sampling_rate),
            SpeechRecognitionDatasetInput(RunningStage.TESTING, test_dataset, sampling_rate=sampling_rate),
            SpeechRecognitionDatasetInput(RunningStage.PREDICTING, predict_dataset, sampling_rate=sampling_rate),
            input_transform=cls.input_transform_cls(
                train_transform, val_transform, test_transform, predict_transform, sampling_rate
            ),
            **data_module_kwargs,
        )
