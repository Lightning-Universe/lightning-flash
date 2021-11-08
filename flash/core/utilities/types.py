from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

from torch import nn
from torchmetrics import Metric

from flash.core.data.io.output import Output
from flash.core.data.io.output_transform import OutputTransform
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.process import Deserializer

MODEL_TYPE = Optional[nn.Module]
LOSS_FN_TYPE = Optional[Union[Callable, Mapping, Sequence]]
OPTIMIZER_TYPE = Union[str, Callable, Tuple[str, Dict[str, Any]]]
LR_SCHEDULER_TYPE = Optional[
    Union[str, Callable, Tuple[str, Dict[str, Any]], Tuple[str, Dict[str, Any], Dict[str, Any]]]
]
METRICS_TYPE = Union[Metric, Mapping, Sequence, None]
DESERIALIZER_TYPE = Optional[Union[Deserializer, Mapping[str, Deserializer]]]
INPUT_TRANSFORM_TYPE= Optional[InputTransform]
OUTPUT_TRANSFORM_TYPE = Optional[OutputTransform]
OUTPUT_TYPE = Optional[Output]
