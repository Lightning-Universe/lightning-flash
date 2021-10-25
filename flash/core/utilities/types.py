from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

from torch import nn
from torchmetrics import Metric

from flash.core.data.process import Deserializer, Postprocess, Preprocess, Serializer

MODEL_TYPE = Optional[nn.Module]
LOSS_FN_TYPE = Optional[Union[Callable, Mapping, Sequence]]
OPTIMIZER_TYPE = Union[str, Callable, Tuple[str, Dict[str, Any]]]
LR_SCHEDULER_TYPE = Optional[
    Union[str, Callable, Tuple[str, Dict[str, Any]], Tuple[str, Dict[str, Any], Dict[str, Any]]]
]
METRICS_TYPE = Union[Metric, Mapping, Sequence, None]
DESERIALIZER_TYPE = Optional[Union[Deserializer, Mapping[str, Deserializer]]]
PREPROCESS_TYPE = Optional[Preprocess]
POSTPROCESS_TYPE = Optional[Postprocess]
SERIALIZER_TYPE = Optional[Union[Serializer, Mapping[str, Serializer]]]
