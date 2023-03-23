from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor

from flash.core.serve.types.base import BaseType


@dataclass(unsafe_hash=True)
class Label(BaseType):
    """Type specifically made for labels that are mapped to a key.

    Parameters
    ----------
    path
        Path to a file that has multiple classes separated by new line character.
        Index of the line will be considered as the key for each class. This parameter
        is mutually exclusive to `classes` parameter
    classes
        A list, tuple or a dict of classes. If it's list or a tuple, index of the
        class, is the key. If it's a dictionary, the key must be an integer
    """

    path: Union[str, Path, None] = field(default=None)
    classes: Union[List, Tuple, Dict, None] = field(default=None, repr=False)

    def __post_init__(self):
        if self.classes is None:
            if self.path is None:
                raise ValueError(
                    "Must provide either classes as a list or " "path to a text file that contains classes"
                )
            with Path(self.path).open(mode="r") as f:
                self.classes = tuple(item.strip() for item in f.readlines())
        if isinstance(self.classes, dict):
            self._reverse_map = {}
            for key, value in self.classes.items():
                if not isinstance(key, int):
                    raise TypeError("Key from the label dict must be an int")
                self._reverse_map[value] = key
        elif isinstance(self.classes, (list, tuple)):
            self._reverse_map = {value: i for i, value in enumerate(self.classes)}
        else:
            raise TypeError("`classes` must be a list, tuple or a dict")

    def deserialize(self, label: str) -> Tensor:
        index = self._reverse_map[label]
        return torch.as_tensor(index)

    def serialize(self, key: Tensor) -> str:
        return self.classes[key.item()]
