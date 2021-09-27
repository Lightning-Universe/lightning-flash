from dataclasses import dataclass
from typing import Callable, Optional

from flash.core.data.properties import ProcessState


@dataclass(unsafe_hash=True, frozen=True)
class PreTensorTransform(ProcessState):

    transform: Optional[Callable] = None


@dataclass(unsafe_hash=True, frozen=True)
class ToTensorTransform(ProcessState):

    transform: Optional[Callable] = None


@dataclass(unsafe_hash=True, frozen=True)
class PostTensorTransform(ProcessState):

    transform: Optional[Callable] = None


@dataclass(unsafe_hash=True, frozen=True)
class CollateFn(ProcessState):

    collate_fn: Optional[Callable] = None
