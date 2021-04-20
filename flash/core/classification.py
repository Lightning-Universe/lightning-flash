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
from typing import Any, Optional

import torch
import torch.nn.functional as F

from flash.core.model import Task
from flash.data.process import Postprocess, Preprocess


class ClassificationPostprocess(Postprocess):

    def __init__(self, multi_label: bool = False, save_path: Optional[str] = None):
        super().__init__(save_path=save_path)
        self.multi_label = multi_label

    def per_sample_transform(self, samples: Any) -> Any:
        if self.multi_label:
            return F.sigmoid(samples).tolist()
        else:
            return torch.argmax(samples, -1).tolist()


class ClassificationTask(Task):

    postprocess_cls = ClassificationPostprocess

    def __init__(self, *args, postprocess: Optional[Preprocess] = None, **kwargs):
        super().__init__(*args, postprocess=postprocess or self.postprocess_cls(), **kwargs)

    def to_metrics_format(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self.hparams, "multi_label", False):
            return F.sigmoid(x).int()
        return F.softmax(x, -1)
