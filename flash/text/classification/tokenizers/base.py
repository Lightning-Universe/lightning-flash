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
from typing import Any, Generator, List, Mapping, Sequence


class BaseTokenizer:
    def __init__(self, backbone: str, pretrained: bool):
        self.backbone = backbone
        self.pretrained = pretrained
        self._is_fit = pretrained

    def fit(self):
        pass

    def _batch_iterator(self, dataset: Sequence[Mapping[str, Any]], input: str) -> Generator[List[str], None, None]:
        for i in range(0, len(dataset), self.batch_size):
            yield dataset[i : i + self.batch_size][input]
