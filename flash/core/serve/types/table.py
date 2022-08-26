from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from flash.core.serve.types.base import BaseType

allowed_types = {
    "float64",
    "float32",
    "float16",
    "complex64",
    "complex128",
    "int64",
    "int32",
    "int16",
    "int8",
    "uint8",
    "bool",
}


@dataclass(unsafe_hash=True)
class Table(BaseType):
    """Table datatype follows the rules of pandas dataframes.

    Pandas dataframe's ``to_dict`` and ``from_dict`` API interface has been used here.
    We rely on pandas exclusively for formatting and conversion to and from dict.
    Also, we offload most of the validations/verifications to pandas. We still do
    few checks explicitly, but with the help of pandas data structure.
    Some of them are:

    *  Length or number of elements in each row (if ``column_names`` provided)
    *  Order of elements (if ``column_names`` are provided)
    *  Invalid data type: Supported dtypes are ``float64``, ``float32``, ``float16``,
       ``complex64``, ``complex128``, ``int64``, ``int32``, ``int16``, ``int8``,
       ``uint8``, and ``bool``

    The layout (orientation) of the incoming/outgoing dictionary is not customizable
    although pandas API allows this. This decision is made to make sure we wouldn't
    have issues handling different layouts in a composition setup downstream.

    Parameters
    ----------
    column_names
        a list of column names to set up in the table.

    Notes
    -----
    *  It might be better to remove pandas dependency to gain performance however we
       are offloading the validation logic to pandas which would have been painful if
       we were to do custom built logic
    """

    column_names: List[str]

    def serialize(self, tensor: Tensor) -> Dict:
        tensor = tensor.numpy()
        df = pd.DataFrame(tensor, columns=self.column_names)
        return df.to_dict()

    def deserialize(self, features: Dict[Union[int, str], Dict[int, Any]]):
        df = pd.DataFrame.from_dict(features)
        if len(self.column_names) != len(df.columns) or not np.all(df.columns == self.column_names):
            raise RuntimeError(
                f"Failed to validate column names. \nExpected: " f"{self.column_names}\nReceived: {list(df.columns)}"
            )
        # TODO: This strict type checking needs to be changed when numpy arrays are returned
        if df.values.dtype.name not in allowed_types:
            raise TypeError(f"Non allowed type {df.values.dtype.name}")
        return torch.from_numpy(df.values)
