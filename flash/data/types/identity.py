from dataclasses import dataclass
from typing import Any

from flash.data.types.base import BaseType


@dataclass(unsafe_hash=True)
class Identity(BaseType):

    def deserialize(self, data: Any) -> Any:
        return data

    def serialize(self, data: Any) -> Any:
        return data
