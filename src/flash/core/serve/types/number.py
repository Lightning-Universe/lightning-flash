from dataclasses import dataclass
from typing import Union

import torch
from torch import Tensor

from flash.core.serve.types.base import BaseType


@dataclass(unsafe_hash=True)
class Number(BaseType):
    """A datatype representing a single item tensor (an int/float number)"""

    def deserialize(self, num: Union[float, int]) -> Tensor:
        return torch.as_tensor(num).view((1, 1))

    def serialize(self, data: Tensor) -> Union[float, int]:
        return data.item()
