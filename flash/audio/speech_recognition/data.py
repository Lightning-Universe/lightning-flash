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
import logging
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import datasets
import pandas as pd
import soundfile as sf
from torch import Tensor

from flash.audio.speech_recognition.collate import DataCollatorCTCWithPadding
from flash.core.data.auto_dataset import AutoDataset
from flash.core.data.data_module import DataModule
from flash.core.data.data_source import DataSource, DefaultDataSources
from flash.core.data.process import Deserializer, Postprocess, Preprocess
from flash.core.utilities.imports import _TEXT_AVAILABLE

if _TEXT_AVAILABLE:
    from datasets import Dataset, load_dataset
    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor


class AudioDeserializer(Deserializer):

    def __init__(self, backbone: str):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(backbone)

    def deserialize(self, sample: Any) -> Tensor:
        return self.tokenizer(sample["speech"], sampling_rate=sample["sampling_rate"][0]).input_values

    @property
    def example_input(self) -> str:
        return "An example input"

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.backbone)


class AudioDataSource(DataSource):

    def __init__(self, backbone: str):
        super().__init__()

        self.backbone = backbone

    def load_data(
        self,
        data: Tuple[str, Union[str, List[str]], Union[str, List[str]]],
        dataset: Optional[Any] = None,
        columns: Union[List[str], Tuple[str]] = ("input_values", "labels"),
    ) -> Union[Sequence[Mapping[str, Any]]]:
        # file, input, target = data
        #
        # data_files = {}
        stage = self.running_stage.value
        # data_files[stage] = str(file)

        dataset_dict = load_dataset("timit_asr")  # todo
        return dataset_dict[stage]

    def predict_load_data(self, data: Any, dataset: AutoDataset):
        return self.load_data(data, dataset, columns=["input_values"])


class SpeechRecognitionPreprocess(Preprocess):

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        backbone: str = "facebook/wav2vec2-base-960h",
    ):
        self.backbone = backbone

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                "timit": AudioDataSource(self.backbone),
            },
            default_data_source=DefaultDataSources.DATASET,
            deserializer=AudioDeserializer(backbone),
        )
        self.processor = Wav2Vec2Processor.from_pretrained(backbone)
        self.collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            **self.transforms,
            "backbone": self.backbone,
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool):
        return cls(**state_dict)

    def _prepare_dataset(self, batch: Any) -> Any:
        # check that all files have the correct sampling rate
        assert (
            len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {self.processor.feature_extractor.sampling_rate}."

        batch["input_values"] = self.processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

        if not self.predicting:
            with self.processor.as_target_processor():
                batch["labels"] = self.processor(batch["target_text"]).input_ids
        return batch

    def _speech_file_to_array_fn(self, batch: Any) -> Any:
        speech_array, sampling_rate = sf.read(batch["file"])
        batch["speech"] = speech_array
        batch["sampling_rate"] = sampling_rate
        if not self.predicting:
            batch["target_text"] = batch["text"]
        return batch

    def _convert_to_batch(self, batch: Any) -> Dataset:
        self._disable_tqdm_bars()
        batch = Dataset.from_pandas(pd.DataFrame(batch))
        columns = ["input_values", "labels"]
        batch = batch.map(partial(self._speech_file_to_array_fn))
        batch = batch.map(partial(self._prepare_dataset), batched=True)
        batch.set_format("torch", columns=columns)
        return batch

    def _disable_tqdm_bars(self):
        datasets.logging.get_verbosity = lambda: logging.NOTSET

    def collate(self, samples: Any) -> Tensor:
        """Override to convert a set of samples to a batch"""
        samples = self._convert_to_batch(samples)
        return self.collator(samples)


class SpeechRecognitionPostprocess(Postprocess):

    def __init__(
        self,
        save_path: Optional[str] = None,
        backbone: str = "facebook/wav2vec2-base-960h",
    ):
        super().__init__(save_path=save_path)
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(backbone)

    def per_batch_transform(self, batch: Any) -> Any:
        transcription = self.tokenizer.batch_decode(batch)[0]
        return transcription


class SpeechRecognitionData(DataModule):
    """Data Module for text classification tasks"""

    preprocess_cls = SpeechRecognitionPreprocess
    postprocess_cls = SpeechRecognitionPostprocess
