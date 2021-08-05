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
from typing import Callable, Dict, Optional, Union

from flash.text.seq2seq.core.data import Seq2SeqData, Seq2SeqPostprocess, Seq2SeqPreprocess


class TranslationPreprocess(Seq2SeqPreprocess):
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        backbone: str = "t5-small",
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = "max_length",
    ):
        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            backbone=backbone,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
        )


class TranslationData(Seq2SeqData):
    """Data module for Translation tasks."""

    preprocess_cls = TranslationPreprocess
    postprocess_cls = Seq2SeqPostprocess
