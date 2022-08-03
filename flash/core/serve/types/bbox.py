from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

from flash.core.serve.types.base import BaseType


@dataclass(unsafe_hash=True)
class BBox(BaseType):
    """Bounding box type to deal with four co-ordinates for object detection tasks.

    Notes
    -----
    *  Although it is explicit and probably familiar to the API consumer systems
       like Javascript to use a dictionary with ``x1, y1, x2 and y2`` as keys, we went
       with DL convention which is to use a list/tuple in which four floats are
       arranged in the same ``order -> x1, y1, x2, y2``
    """

    def __post_init__(self):
        self._valid_size = torch.Size([4])

    def _validate(self, elem):
        if elem.shape != self._valid_size:
            raise ValueError("Each box must consist of (only) four elements each " "corresponding to x1, x2, y1 and y2")
        if elem.dtype == torch.bool or torch.is_complex(elem):
            raise TypeError(f"Found unsupported datatype for " f"bounding boxes: {elem.dtype}")

    def deserialize(self, box: Tuple[float, ...]) -> Tensor:
        tensor = torch.FloatTensor(box)
        self._validate(tensor)
        return tensor

    def serialize(self, box: Tensor) -> Tuple[float, ...]:
        box = box.squeeze()
        self._validate(box)
        return box.tolist()
