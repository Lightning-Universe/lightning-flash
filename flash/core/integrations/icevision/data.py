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
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np

from flash.core.data.io.classification_input import ClassificationState
from flash.core.data.io.input import DataKeys
from flash.core.data.io.input_base import Input
from flash.core.data.utilities.paths import list_valid_files
from flash.core.integrations.icevision.transforms import from_icevision_record
from flash.core.utilities.imports import _ICEVISION_AVAILABLE
from flash.image.data import image_loader, IMG_EXTENSIONS, NP_EXTENSIONS

if _ICEVISION_AVAILABLE:
    from icevision.core.record import BaseRecord
    from icevision.core.record_components import ClassMapRecordComponent, FilepathRecordComponent, tasks
    from icevision.data.data_splitter import SingleSplitSplitter
    from icevision.parsers.parser import Parser


class IceVisionInput(Input):
    def load_data(
        self,
        root: str,
        ann_file: Optional[str] = None,
        parser: Optional[Type["Parser"]] = None,
    ) -> List[Dict[str, Any]]:
        if inspect.isclass(parser) and issubclass(parser, Parser):
            parser = parser(ann_file, root)
        elif isinstance(parser, Callable):
            parser = parser(root)
        else:
            raise ValueError("The parser must be a callable or an IceVision Parser type.")
        self.num_classes = parser.class_map.num_classes
        self.set_state(ClassificationState([parser.class_map.get_by_id(i) for i in range(self.num_classes)]))
        records = parser.parse(data_splitter=SingleSplitSplitter())
        return [{DataKeys.INPUT: record} for record in records[0]]

    def predict_load_data(
        self, paths: Union[str, List[str]], ann_file: Optional[str] = None, parser: Optional[Type["Parser"]] = None
    ) -> List[Dict[str, Any]]:
        if parser is not None:
            return self.load_data(paths, ann_file, parser)
        paths = list_valid_files(paths, valid_extensions=IMG_EXTENSIONS + NP_EXTENSIONS)
        return [{DataKeys.INPUT: path} for path in paths]

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        record = sample[DataKeys.INPUT].load()
        return from_icevision_record(record)

    def predict_load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(sample[DataKeys.INPUT], BaseRecord):
            return self.load_sample(sample)
        filepath = sample[DataKeys.INPUT]
        image = np.array(image_loader(filepath))

        record = BaseRecord([FilepathRecordComponent()])
        record.filepath = filepath
        record.set_img(image)
        record.add_component(ClassMapRecordComponent(task=tasks.detection))
        return from_icevision_record(record)
