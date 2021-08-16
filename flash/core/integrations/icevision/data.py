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
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type

import numpy as np

from flash.core.data.data_source import DefaultDataKeys
from flash.core.integrations.icevision.transforms import from_icevision_record
from flash.core.utilities.imports import _ICEVISION_AVAILABLE
from flash.image.data import ImagePathsDataSource

if _ICEVISION_AVAILABLE:
    from icevision.core.record import BaseRecord
    from icevision.core.record_components import ClassMapRecordComponent, ImageRecordComponent, tasks
    from icevision.data.data_splitter import SingleSplitSplitter
    from icevision.parsers.parser import Parser


class IceVisionPathsDataSource(ImagePathsDataSource):
    def predict_load_data(self, data: Tuple[str, str], dataset: Optional[Any] = None) -> Sequence[Dict[str, Any]]:
        return super().predict_load_data(data, dataset)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        record = sample[DefaultDataKeys.INPUT].load()
        return from_icevision_record(record)

    def predict_load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample = super().load_sample(sample)
        image = np.array(sample[DefaultDataKeys.INPUT])
        record = BaseRecord([ImageRecordComponent()])

        record.set_img(image)
        record.add_component(ClassMapRecordComponent(task=tasks.detection))
        return from_icevision_record(record)


class IceVisionParserDataSource(IceVisionPathsDataSource):
    def __init__(self, parser: Optional[Type["Parser"]] = None):
        super().__init__()
        self.parser = parser

    def load_data(self, data: Tuple[str, str], dataset: Optional[Any] = None) -> Sequence[Dict[str, Any]]:
        root, ann_file = data

        if self.parser is not None:
            parser = self.parser(ann_file, root)
            dataset.num_classes = len(parser.class_map)
            records = parser.parse(data_splitter=SingleSplitSplitter())
            return [{DefaultDataKeys.INPUT: record} for record in records[0]]
        else:
            raise ValueError("The parser type must be provided")


class IceDataParserDataSource(IceVisionPathsDataSource):
    def __init__(self, parser: Optional[Callable] = None):
        super().__init__()
        self.parser = parser

    def load_data(self, data: Tuple[str, str], dataset: Optional[Any] = None) -> Sequence[Dict[str, Any]]:
        root = data

        if self.parser is not None:
            parser = self.parser(root)
            dataset.num_classes = len(parser.class_map)
            records = parser.parse(data_splitter=SingleSplitSplitter())
            return [{DefaultDataKeys.INPUT: record} for record in records[0]]
        else:
            raise ValueError("The parser must be provided")
