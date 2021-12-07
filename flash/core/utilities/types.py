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
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type, Union

from pytorch_lightning.utilities.enums import LightningEnum
from torch import nn
from torchmetrics import Metric

import flash
from flash.core.data.io.output import Output

# Task Arguments
MODEL_TYPE = Optional[nn.Module]
LOSS_FN_TYPE = Optional[Union[Callable, Mapping, Sequence]]
OPTIMIZER_TYPE = Union[str, Callable, Tuple[str, Dict[str, Any]]]
LR_SCHEDULER_TYPE = Optional[
    Union[str, Callable, Tuple[str, Dict[str, Any]], Tuple[str, Dict[str, Any], Dict[str, Any]]]
]
METRICS_TYPE = Union[Metric, Mapping, Sequence, None]

# Data Pipeline
DESERIALIZER_TYPE = Optional[
    Union["flash.core.data.process.Deserializer", Mapping[str, "flash.core.data.process.Deserializer"]]
]
INPUT_TRANSFORM_TYPE = Optional[
    Union[
        Type["flash.InputTransform"],
        Callable,
        Tuple[Union[LightningEnum, str], Dict[str, Any]],
        Union[LightningEnum, str],
    ]
]

OUTPUT_TRANSFORM_TYPE = Optional["flash.OutputTransform"]
OUTPUT_TYPE = Optional[Output]
