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
import collections
from typing import Any

from flash.core.data.io.output_transform import OutputTransform
from flash.core.utilities.imports import requires


class QuestionAnsweringOutputTransform(OutputTransform):
    @requires("text")
    def uncollate(self, predicted_sentences: collections.OrderedDict) -> Any:
        uncollated_predicted_sentences = []
        for key in predicted_sentences:
            uncollated_predicted_sentences.append({key: predicted_sentences[key]})
        return uncollated_predicted_sentences
