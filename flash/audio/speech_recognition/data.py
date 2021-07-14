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
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Union
from typing import Callable, Mapping, Sequence, Tuple

import soundfile as sf
import torch
from torch import Tensor

from flash.core.data.auto_dataset import AutoDataset
from flash.core.data.data_module import DataModule
from flash.core.data.data_source import DataSource, DefaultDataSources
from flash.core.data.process import Deserializer, Postprocess, Preprocess
from flash.core.utilities.imports import _TEXT_AVAILABLE

if _TEXT_AVAILABLE:
    from datasets import load_dataset
    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor


class AudioDeserializer(Deserializer):

    def __init__(self, backbone: str, max_length: int):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(backbone)
        self.max_length = max_length

    def deserialize(self, sample: Any) -> Tensor:
        return self.tokenizer(sample["speech"], sampling_rate=sample["sampling_rate"][0]).input_values

    @property
    def example_input(self) -> str:
        return "An example input"

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("processor")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.backbone)


class AudioDataSource(DataSource):

    def __init__(self, backbone: str, max_length: int = 128):
        super().__init__()

        self.backbone = backbone
        self.processor = Wav2Vec2Processor.from_pretrained(backbone)
        self.max_length = max_length

    def _prepare_dataset(self, batch):
        # check that all files have the correct sampling rate
        assert (
            len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {self.processor.feature_extractor.sampling_rate}."

        batch["input_values"] = self.processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

        if not self.predicting:
            with self.processor.as_target_processor():
                batch["labels"] = self.processor(batch["target_text"]).input_ids
        return batch

    @staticmethod
    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = sf.read(batch["file"])
        batch["speech"] = speech_array
        batch["sampling_rate"] = sampling_rate
        batch["target_text"] = batch["text"]
        return batch

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
        dataset_dict = dataset_dict.map(AudioDataSource.speech_file_to_array_fn, num_proc=4)
        dataset_dict = dataset_dict.map(partial(self._prepare_dataset), batched=True)
        dataset_dict.set_format("torch", columns=columns)

        return dataset_dict[stage]

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = Wav2Vec2Processor.from_pretrained(self.backbone)

    def predict_load_data(self, data: Any, dataset: AutoDataset):
        return self.load_data(data, dataset, columns=["input_values"])


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


class SpeechRecognitionPreprocess(Preprocess):

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        backbone: str = "facebook/wav2vec2-base-960h",
        max_length: int = 128,
    ):
        self.backbone = backbone
        self.max_length = max_length

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                "timit": AudioDataSource(self.backbone, max_length=max_length),
            },
            default_data_source=DefaultDataSources.DATASET,
            deserializer=AudioDeserializer(backbone, max_length),
        )
        self.processor = Wav2Vec2Processor.from_pretrained(backbone)
        self.collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            **self.transforms,
            "backbone": self.backbone,
            "max_length": self.max_length,
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool):
        return cls(**state_dict)

    def collate(self, samples: Any) -> Tensor:
        """Override to convert a set of samples to a batch"""
        # data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

        if isinstance(samples, dict):
            samples = [samples]
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
