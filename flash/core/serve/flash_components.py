import inspect
from pathlib import Path
from typing import Any, Callable, Optional, Type

import torch
from pytorch_lightning.trainer.states import RunningStage

from flash import Task
from flash.core.serve import Composition, expose, GridModel, ModelComponent
from flash.core.serve.core import FilePath, GridModelValidArgs_T, GridserveScriptLoader
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

    def serialize(self, output) -> Any:  # pragma: no cover
        result = self._serializer(output)
        return result

    def deserialize(self, data: str) -> Any:  # pragma: no cover
        return None


class FlashServeScriptLoader(GridserveScriptLoader):

    model_cls: Optional[Task] = None

    def __init__(self, location: FilePath):
        self.location = location
        self.instance = self.model_cls.load_from_checkpoint(location)
