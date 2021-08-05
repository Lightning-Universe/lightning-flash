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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset

import flash
from flash.core.data.data_module import DataModule
from flash.core.data.data_source import (
    DatasetDataSource,
    DataSource,
    DefaultDataKeys,
    DefaultDataSources,
    PathsDataSource,
)
from flash.core.data.process import Deserializer, Postprocess, Preprocess
from flash.core.data.properties import ProcessState
from flash.core.utilities.imports import _AUDIO_AVAILABLE, requires_extras

if _AUDIO_AVAILABLE:
    import soundfile as sf
    from datasets import Dataset as HFDataset
    from datasets import load_dataset
    from transformers import Wav2Vec2CTCTokenizer
else:
    HFDataset = object


class SpeechRecognitionDeserializer(Deserializer):
    def deserialize(self, sample: Any) -> Dict:
        encoded_with_padding = (sample + "===").encode("ascii")
        audio = base64.b64decode(encoded_with_padding)
        buffer = io.BytesIO(audio)
        data, sampling_rate = sf.read(buffer)
        return {
            DefaultDataKeys.INPUT: data,
            DefaultDataKeys.METADATA: {"sampling_rate": sampling_rate},
        }

    @property
    def example_input(self) -> str:
        with (Path(flash.ASSETS_ROOT) / "example.wav").open("rb") as f:
            return base64.b64encode(f.read()).decode("UTF-8")


class BaseSpeechRecognition:
    def _load_sample(self, sample: Dict[str, Any]) -> Any:
        path = sample[DefaultDataKeys.INPUT]
        if (
            not os.path.isabs(path)
            and DefaultDataKeys.METADATA in sample
            and "root" in sample[DefaultDataKeys.METADATA]
        ):
            path = os.path.join(sample[DefaultDataKeys.METADATA]["root"], path)
        speech_array, sampling_rate = sf.read(path)
        sample[DefaultDataKeys.INPUT] = speech_array
        sample[DefaultDataKeys.METADATA] = {"sampling_rate": sampling_rate}
        return sample


class SpeechRecognitionFileDataSource(DataSource, BaseSpeechRecognition):
    def __init__(self, filetype: Optional[str] = None):
        super().__init__()
        self.filetype = filetype

    def load_data(
        self,
        data: Tuple[str, Union[str, List[str]], Union[str, List[str]]],
        dataset: Optional[Any] = None,
    ) -> Union[Sequence[Mapping[str, Any]]]:
        if self.filetype == "json":
            file, input_key, target_key, field = data
        else:
            file, input_key, target_key = data
        stage = self.running_stage.value
        if self.filetype == "json" and field is not None:
            dataset_dict = load_dataset(self.filetype, data_files={stage: str(file)}, field=field)
        else:
            dataset_dict = load_dataset(self.filetype, data_files={stage: str(file)})

        dataset = dataset_dict[stage]
        meta = {"root": os.path.dirname(file)}
        return [
            {
                DefaultDataKeys.INPUT: input_file,
                DefaultDataKeys.TARGET: target,
                DefaultDataKeys.METADATA: meta,
            }
            for input_file, target in zip(dataset[input_key], dataset[target_key])
        ]

    def load_sample(self, sample: Dict[str, Any], dataset: Any = None) -> Any:
        return self._load_sample(sample)


class SpeechRecognitionCSVDataSource(SpeechRecognitionFileDataSource):
    def __init__(self):
        super().__init__(filetype="csv")


class SpeechRecognitionJSONDataSource(SpeechRecognitionFileDataSource):
    def __init__(self):
        super().__init__(filetype="json")


class SpeechRecognitionDatasetDataSource(DatasetDataSource, BaseSpeechRecognition):
    def load_data(self, data: Dataset, dataset: Optional[Any] = None) -> Union[Sequence[Mapping[str, Any]]]:
        if isinstance(data, HFDataset):
            data = list(zip(data["file"], data["text"]))
        return super().load_data(data, dataset)


class SpeechRecognitionPathsDataSource(PathsDataSource, BaseSpeechRecognition):
    def __init__(self):
        super().__init__(("wav", "ogg", "flac", "mat"))

    def load_sample(self, sample: Dict[str, Any], dataset: Any = None) -> Any:
        return self._load_sample(sample)


class SpeechRecognitionPreprocess(Preprocess):
    @requires_extras("audio")
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
    ):
        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.CSV: SpeechRecognitionCSVDataSource(),
                DefaultDataSources.JSON: SpeechRecognitionJSONDataSource(),
                DefaultDataSources.FILES: SpeechRecognitionPathsDataSource(),
                DefaultDataSources.DATASETS: SpeechRecognitionDatasetDataSource(),
            },
            default_data_source=DefaultDataSources.FILES,
            deserializer=SpeechRecognitionDeserializer(),
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return self.transforms

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(**state_dict)


@dataclass(unsafe_hash=True, frozen=True)
class SpeechRecognitionBackboneState(ProcessState):
    """The ``SpeechRecognitionBackboneState`` stores the backbone in use by the
    :class:`~flash.audio.speech_recognition.data.SpeechRecognitionPostprocess`
    """

    backbone: str


class SpeechRecognitionPostprocess(Postprocess):
    @requires_extras("audio")
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

    preprocess_cls = SpeechRecognitionPreprocess
    postprocess_cls = SpeechRecognitionPostprocess
