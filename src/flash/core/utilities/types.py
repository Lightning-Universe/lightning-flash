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
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Type, TypeVar, Union

from pytorch_lightning.utilities.enums import LightningEnum
from torch import nn
from torchmetrics import Metric

import flash

# Task Arguments
MODEL_TYPE = TypeVar("MODEL_TYPE", nn.Module, None)
LOSS_FN_TYPE = TypeVar("LOSS_FN_TYPE", Callable, Mapping, Sequence, None)
OPTIMIZER_TYPE = TypeVar("OPTIMIZER_TYPE", str, Callable, Tuple[str, Dict[str, Any]], None)
LR_SCHEDULER_TYPE = TypeVar(
    "LR_SCHEDULER_TYPE", str, Callable, Tuple[str, Dict[str, Any]], Tuple[str, Dict[str, Any], Dict[str, Any]], None
)
METRICS_TYPE = TypeVar("METRICS_TYPE", Metric, Mapping, Sequence, None)

# Data Pipeline
INPUT_TRANSFORM_TYPE = TypeVar(
    "INPUT_TRANSFORM_TYPE",
    Type["flash.core.data.io.input_transform.InputTransform"],
    Callable,
    Tuple[Union[LightningEnum, str], Dict[str, Any]],
    Union[LightningEnum, str],
    None,
)
OUTPUT_TRANSFORM_TYPE = TypeVar("OUTPUT_TRANSFORM_TYPE", "flash.core.data.io.output_transform.OutputTransform", None)
OUTPUT_TYPE = TypeVar("OUTPUT_TYPE", "flash.core.data.io.output.Output", None)
