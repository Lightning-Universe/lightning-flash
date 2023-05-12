from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

from torch import Tensor

from flash.core.serve.types.base import BaseType


@dataclass(unsafe_hash=True)
class Repeated(BaseType):
    """Allow repeated specification of some dtype.

    Attributes
    ----------
    dtype:
        Data type of the repeated object.
    max_len:
        Optional parameter specifying if there is a maximum length of the
        repeated elements (`int > 0`). If `max_len=None`, there can be any
         number of repeated elements. By default: `None`.
    """

    dtype: BaseType
    max_len: Optional[int] = field(default=None)

    @property
    def type_hints(self):
        """Fetch hints from ``dtype`` attr and make it available for ``EndpointProtocol``."""
        _type_hints = getattr(self, "_type_hints", None)
        if not _type_hints:
            _type_hints = {
                "output_args": self.dtype.type_hints["output_args"],
                "input_args": self.dtype.type_hints["input_args"],
            }
            setattr(self, "_type_hints", _type_hints)
        return _type_hints

    def __post_init__(self):
        if not isinstance(self.dtype, BaseType):
            raise TypeError(f"dtype argument must inherit from {BaseType}")
        if isinstance(self.dtype, type(self)):
            raise TypeError(f"cannot specify {type(self)} as dtype of {type(self)}")

        if self.max_len is not None:
            if not isinstance(self.max_len, int):
                raise TypeError(f"`max_len` must be {int}, not {type(self.max_len)}")
            if self.max_len <= 0:
                raise ValueError(f"`max_len={self.max_len}` is not >= 1.")

    def deserialize(self, *args: Dict) -> Tuple[Tensor, ...]:
        if (self.max_len is not None) and (len(args) > self.max_len):
            raise ValueError(f"len(arg)={len(args)} > self.max_len={self.max_len}")
        return tuple(self.dtype.deserialize(**item) for item in args)

    def packed_deserialize(self, args):
        """Arguments are positional arguments for deserialize, unlike other datatypes."""
        return self.deserialize(*args)

    def serialize(self, args: Sequence) -> Tuple[Any, ...]:
        if (self.max_len is not None) and (len(args) > self.max_len):
            raise ValueError(f"len(arg)={len(args)} > self.max_len={self.max_len}")
        return tuple(self.dtype.serialize(item) for item in args)
