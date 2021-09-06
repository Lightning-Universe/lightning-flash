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
import inspect
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type

import numpy as np

from flash.core.data.data_source import DefaultDataKeys, LabelsState
from flash.core.integrations.icevision.transforms import from_icevision_record
from flash.core.utilities.imports import _ICEVISION_AVAILABLE
from flash.image.data import ImagePathsDataSource

if _ICEVISION_AVAILABLE:
    from icevision.core.record import BaseRecord
    from icevision.core.record_components import ClassMapRecordComponent, FilepathRecordComponent, tasks
    from icevision.data.data_splitter import SingleSplitSplitter
    from icevision.parsers.parser import Parser


class IceVisionPathsDataSource(ImagePathsDataSource):
    def predict_load_data(self, data: Tuple[str, str], dataset: Optional[Any] = None) -> Sequence[Dict[str, Any]]:
        return super().predict_load_data(data, dataset)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        record = sample[DefaultDataKeys.INPUT].load()
        return from_icevision_record(record)

    def predict_load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(sample[DefaultDataKeys.INPUT], BaseRecord):
            return self.load_sample(sample)
        filepath = sample[DefaultDataKeys.INPUT]
        sample = super().load_sample(sample)
        image = np.array(sample[DefaultDataKeys.INPUT])

        record = BaseRecord([FilepathRecordComponent()])
        record.filepath = filepath
        record.set_img(image)
        record.add_component(ClassMapRecordComponent(task=tasks.detection))
        return from_icevision_record(record)


class IceVisionParserDataSource(IceVisionPathsDataSource):
    def __init__(self, parser: Optional[Type["Parser"]] = None):
        super().__init__()
        self.parser = parser

    def load_data(self, data: Tuple[str, str], dataset: Optional[Any] = None) -> Sequence[Dict[str, Any]]:
        if self.parser is not None:
            if inspect.isclass(self.parser) and issubclass(self.parser, Parser):
                root, ann_file = data
                parser = self.parser(ann_file, root)
            elif isinstance(self.parser, Callable):
                parser = self.parser(data)
            else:
                raise ValueError("The parser must be a callable or an IceVision Parser type.")
            dataset.num_classes = parser.class_map.num_classes
            self.set_state(LabelsState([parser.class_map.get_by_id(i) for i in range(dataset.num_classes)]))
            records = parser.parse(data_splitter=SingleSplitSplitter())
            return [{DefaultDataKeys.INPUT: record} for record in records[0]]
        else:
            raise ValueError("The parser argument must be provided.")

    def predict_load_data(self, data: Any, dataset: Optional[Any] = None) -> Sequence[Dict[str, Any]]:
        result = super().predict_load_data(data, dataset)
        if len(result) == 0:
            result = self.load_data(data, dataset)
        return result
