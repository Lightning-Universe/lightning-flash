from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

import numpy as np
import torch

try:
    from PIL import Image as PILImage
except (ImportError, ModuleNotFoundError) as e:  # pragma: no cover
    msg = f"install the 'Pillow' package to make use of this feature"
    raise ValueError(msg) from e

from flash.data.types.base import BaseType


@dataclass(unsafe_hash=True)
class Identity(BaseType):

    def deserialize(self, data: Any) -> Any:
        return data

    def serialize(self, data: Any) -> Any:
        import pdb
        pdb.set_trace()
        return data
