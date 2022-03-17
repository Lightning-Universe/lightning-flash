# Copyright 2020 The PyTorch Lightning team and The HuggingFace Team. All rights reserved.

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
from typing import Any, Callable, Dict, List, Optional, Union

from flash.core.data.io.input import DataKeys
from flash.core.data.io.input_transform import InputTransform
from flash.core.utilities.imports import _AUDIO_AVAILABLE

if _AUDIO_AVAILABLE:
    from transformers import AutoProcessor
else:
    AutoProcessor = object


@dataclass
class SpeechRecognitionInputCollateTransform(InputTransform):

    backbone: str = "facebook/wav2vec2-base-960h"
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __post_init__(self):
        self.processor: AutoProcessor = AutoProcessor.from_pretrained(self.backbone)
        super().__post_init__()

    def collate(self) -> Callable:
        """Defines the transform to be applied on a list of sample to create a batch for all stages."""

        def data_collator_ctc_with_padding(samples: List[Dict[str, Any]], metadata: List[Dict[str, Any]]):
            inputs = [sample[DataKeys.INPUT] for sample in samples]
            sampling_rates = [sample["sampling_rate"] for sample in metadata]

            assert (
                len(set(sampling_rates)) == 1
            ), f"Make sure all inputs have the same sampling rate of {self.processor.feature_extractor.sampling_rate}."

            inputs = self.processor(inputs, sampling_rate=sampling_rates[0]).input_values

            # split inputs and labels since they have to be of different lengths and need
            # different padding methods
            input_features = [{"input_values": input} for input in inputs]

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            labels = [sample.get(DataKeys.TARGET, None) for sample in samples]
            # check to ensure labels exist to collate
            if None not in labels:
                with self.processor.as_target_processor():
                    label_features = self.processor(labels).input_ids
                    label_features = [{"input_ids": feature} for feature in label_features]
                    labels_batch = self.processor.pad(
                        label_features,
                        padding=self.padding,
                        max_length=self.max_length_labels,
                        pad_to_multiple_of=self.pad_to_multiple_of_labels,
                        return_tensors="pt",
                    )

                # replace padding with -100 to ignore loss correctly
                batch["labels"] = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            return batch

        return data_collator_ctc_with_padding
