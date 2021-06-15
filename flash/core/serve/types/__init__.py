import importlib

from flash.core.serve.types.base import BaseType
from flash.core.serve.types.bbox import BBox
from flash.core.serve.types.image import Image
from flash.core.serve.types.label import Label
from flash.core.serve.types.number import Number
from flash.core.serve.types.repeated import Repeated
from flash.core.serve.types.table import Table
from flash.core.serve.types.text import Text

__all__ = ("BaseType", "Number", "Image", "Text", "Label", "Table", "BBox", "Repeated")


def __getattr__(name: str):
    if name in __all__:
        return getattr(importlib.import_module(f".{name.lower()}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
