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

from flash.core.data.io.input import DataKeys, Input
from flash.core.data.utilities.loading import IMG_EXTENSIONS, load_image, NP_EXTENSIONS
from flash.core.data.utilities.paths import list_valid_files
from flash.core.integrations.icevision.transforms import from_icevision_record
from flash.core.utilities.imports import _ICEVISION_AVAILABLE, requires

if _ICEVISION_AVAILABLE:
    from icevision.core.record import BaseRecord
    from icevision.core.record_components import ClassMapRecordComponent, FilepathRecordComponent, tasks
    from icevision.data.data_splitter import SingleSplitSplitter
    from icevision.parsers.parser import Parser


class IceVisionInput(Input):
    num_classes: int
    labels: list

    @requires("icevision")
    def load_data(
        self,
        root: str,
        ann_file: Optional[str] = None,
        parser: Optional[Type["Parser"]] = None,
        parser_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        parser_kwargs = {} if parser_kwargs is None else parser_kwargs
        unwrapped_parser = getattr(parser, "func", parser)
        if inspect.isclass(unwrapped_parser) and issubclass(unwrapped_parser, Parser):
            parser = parser(ann_file, root, **parser_kwargs)
        elif isinstance(unwrapped_parser, Callable):
            parser = parser(root, **parser_kwargs)
        else:
            raise ValueError("The parser must be a callable or an IceVision Parser type.")
        class_map = getattr(parser, "class_map", None)
        if class_map is not None:
            self.num_classes = class_map.num_classes
            self.labels = [class_map.get_by_id(i) for i in range(self.num_classes)]
        records = parser.parse(data_splitter=SingleSplitSplitter())
        return [{DataKeys.INPUT: record} for record in records[0]]

    def predict_load_data(
        self, paths: Union[str, List[str]], parser: Optional[Type["Parser"]] = None
    ) -> List[Dict[str, Any]]:
        paths = list_valid_files(paths, valid_extensions=IMG_EXTENSIONS + NP_EXTENSIONS)
        return [{DataKeys.INPUT: path} for path in paths]

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        record = sample[DataKeys.INPUT].load()
        return from_icevision_record(record)

    def predict_load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(sample[DataKeys.INPUT], BaseRecord):
            return self.load_sample(sample)
        filepath = sample[DataKeys.INPUT]
        image = np.array(load_image(filepath))

        record = BaseRecord([FilepathRecordComponent()])
        record.filepath = filepath
        record.set_img(image)
        record.add_component(ClassMapRecordComponent(task=tasks.detection))
        return from_icevision_record(record)
