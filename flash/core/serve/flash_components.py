from typing import Any, Callable, Mapping, Optional

import torch

from flash import Task
from flash.core.data.data_source import DefaultDataKeys
from flash.core.serve.core import FilePath, GridserveScriptLoader
from flash.core.serve.types.base import BaseType


class FlashInputs(BaseType):

    def __init__(
        self,
        deserializer: Callable,
    ):
        self._deserializer = deserializer

    def serialize(self, *args) -> Any:  # pragma: no cover
        return None

    def deserialize(self, data: str) -> Any:  # pragma: no cover
        return self._deserializer(data)


class FlashOutputs(BaseType):

    def __init__(
        self,
        serializer: Callable,
    ):
        self._serializer = serializer

    def serialize(self, outputs) -> Any:  # pragma: no cover
        results = []
        if isinstance(outputs, list) or isinstance(outputs, torch.Tensor):
            for output in outputs:
                result = self._serializer(output)
                if isinstance(result, Mapping):
                    result = result[DefaultDataKeys.PREDS]
                results.append(result)
        if len(results) == 1:
            return results[0]
        return results

    def deserialize(self, data: str) -> Any:  # pragma: no cover
        return None


class FlashServeScriptLoader(GridserveScriptLoader):

    model_cls: Optional[Task] = None

    def __init__(self, location: FilePath):
        self.location = location
        self.instance = self.model_cls.load_from_checkpoint(location)
