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
from typing import Any, Dict, List, Optional

from flash.core.data.data_module import DataModule
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.io.output_transform import OutputTransform


class TabularData(DataModule):

    input_transform_cls = InputTransform
    output_transform_cls = OutputTransform

    @property
    def parameters(self) -> Optional[Dict[str, Any]]:
        """The parameters dictionary created from the train data when constructing the ``TabularData`` object."""
        return getattr(self.train_dataset, "parameters", None)

    @property
    def codes(self) -> Dict[str, str]:
        return self.parameters["codes"]

    @property
    def categorical_fields(self) -> Optional[List[str]]:
        return self.parameters["categorical_fields"]

    @property
    def numerical_fields(self) -> Optional[List[str]]:
        return self.parameters["numerical_fields"]

    @property
    def num_features(self) -> int:
        return len(self.categorical_fields) + len(self.numerical_fields)

    @property
    def cat_dims(self) -> list:
        return [len(self.codes[cat]) + 1 for cat in self.categorical_fields]

    @property
    def embedding_sizes(self) -> list:
        """Recommended embedding sizes."""

        # https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
        # The following "formula" provides a general rule of thumb about the number of embedding dimensions:
        # embedding_dimensions =  number_of_categories**0.25
        emb_dims = [max(int(n**0.25), 16) for n in self.cat_dims]
        return list(zip(self.cat_dims, emb_dims))
