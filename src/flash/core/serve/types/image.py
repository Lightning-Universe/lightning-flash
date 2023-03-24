import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from flash.core.utilities.imports import _PIL_AVAILABLE

if _PIL_AVAILABLE:
    from PIL import Image as PILImage

from flash.core.serve.types.base import BaseType


@dataclass(unsafe_hash=True)
class Image(BaseType):
    """Image output.

    Notes
    -----
    *  The ``modes`` parameter can take on any one of the following values:

       .. code-block:: python

        {
            1: 1,  # (1-bit pixels, black and white, stored with one pixel per byte)
            "L": 1,  # (8-bit pixels, black and white)
            "P": 1,  # (8-bit pixels, mapped to any other mode using a color palette)
            "RGB": 3,  # (3x8-bit pixels, true color)
            "RGBX": 4,  # RGB with padding
            "RGBA": 4,  # (4x8-bit pixels, true color with transparency mask)
            "RGBa": 3,  # (3x8-bit pixels, true color with pre-multiplied alpha)
            "CMYK": 4,  # (4x8-bit pixels, color separation)
            "YCbCr": 3,  # (3x8-bit pixels, color video format)
            "LAB": 3,  # (3x8-bit pixels, the L*a*b color space)
            "HSV": 3,  # (3x8-bit pixels, Hue, Saturation, Value color space)
            "I": 1,  # (32-bit signed integer pixels)
            "F": 1,  # (32-bit floating point pixels)
        }
    """

    height: Optional[int] = None
    width: Optional[int] = None
    extension: str = "JPEG"
    mode: str = "RGB"
    channel_first: bool = False

    def deserialize(self, data: str) -> Tensor:
        encoded_with_padding = (data + "===").encode("ascii")
        img = base64.b64decode(encoded_with_padding)
        buffer = BytesIO(img)
        img = PILImage.open(buffer, mode="r")
        if self.height and self.width:
            img = img.resize((self.width, self.height))
        arr = np.array(img)
        # TODO: add batch dimension based on the argument
        return torch.from_numpy(arr).unsqueeze(0)

    def serialize(self, tensor: Tensor) -> str:
        tensor = tensor.squeeze(0).numpy()
        image = PILImage.fromarray(tensor)
        if image.mode != self.mode:
            image = image.convert(self.mode)
        buffer = BytesIO()
        image.save(buffer, format=self.extension.lower())
        buffer.seek(0)
        encoded = buffer.getvalue()
        return base64.b64encode(encoded).decode("ascii")
