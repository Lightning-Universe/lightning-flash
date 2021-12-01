from dataclasses import dataclass
from typing import Callable, Optional

from flash.core.data.properties import ProcessState


@dataclass(unsafe_hash=True, frozen=True)
class PerSampleTransform(ProcessState):

    transform: Optional[Callable] = None


@dataclass(unsafe_hash=True, frozen=True)
class PerSampleTransformOnDevice(ProcessState):

    transform: Optional[Callable] = None


@dataclass(unsafe_hash=True, frozen=True)
class PerBatchTransform(ProcessState):

    transform: Optional[Callable] = None


@dataclass(unsafe_hash=True, frozen=True)
class PerBatchTransformOnDevice(ProcessState):

    transform: Optional[Callable] = None


@dataclass(unsafe_hash=True, frozen=True)
class CollateFn(ProcessState):

    collate_fn: Optional[Callable] = None
