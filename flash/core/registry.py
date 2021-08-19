# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
from typing import Any, Callable, Dict, List, Optional, Union

from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.core.utilities.providers import Provider

_REGISTERED_FUNCTION = Dict[str, Any]


def print_provider_info(name, providers, func):
    if not isinstance(providers, List):
        providers = [providers]
    providers = list(providers)
    if len(providers) > 1:
        providers[-2] = f"{str(providers[-2])} and {str(providers[-1])}"
        providers = providers[:-1]
    message = f"Using '{name}' provided by {', '.join(str(provider) for provider in providers)}."

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank_zero_info(message)
        return func(*args, **kwargs)

    return wrapper


class FlashRegistry:
    """This class is used to register function or :class:`functools.partial` class to a registry."""

    def __init__(self, name: str, verbose: bool = False) -> None:
        self.name = name
        self.functions: List[_REGISTERED_FUNCTION] = []
        self._verbose = verbose

    def __len__(self) -> int:
        return len(self.functions)

    def __contains__(self, key) -> bool:
        return any(key == e["name"] for e in self.functions)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, functions={self.functions})"

    def get(
        self,
        key: str,
        with_metadata: bool = False,
        strict: bool = True,
        **metadata,
    ) -> Union[Callable, _REGISTERED_FUNCTION, List[_REGISTERED_FUNCTION], List[Callable]]:
        """This function is used to gather matches from the registry:

        Args:
            key: Name of the registered function.
            with_metadata: Whether to include the associated metadata in the return value.
            strict: Whether to return all matches or just one.
            metadata: Metadata used to filter against existing registry item's metadata.
        """
        matches = [e for e in self.functions if key == e["name"]]
        if not matches:
            raise KeyError(f"Key: {key} is not in {type(self).__name__}")

        if metadata:
            matches = [m for m in matches if metadata.items() <= m["metadata"].items()]
            if not matches:
                raise KeyError("Found no matches that fit your metadata criteria. Try removing some metadata")

        matches = [e if with_metadata else e["fn"] for e in matches]
        return matches[0] if strict else matches

    def remove(self, key: str) -> None:
        self.functions = [f for f in self.functions if f["name"] != key]

    def _register_function(
        self,
        fn: Callable,
        name: Optional[str] = None,
        override: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not callable(fn):
            raise MisconfigurationException(f"You can only register a callable, found: {fn}")

        name = name or fn.__name__

        if self._verbose:
            rank_zero_info(f"Registering: {fn.__name__} function with name: {name} and metadata: {metadata}")

        if "providers" in metadata:
            providers = metadata["providers"]
            fn = print_provider_info(name, providers, fn)

        item = {"fn": fn, "name": name, "metadata": metadata or {}}

        matching_index = self._find_matching_index(item)
        if override and matching_index is not None:
            self.functions[matching_index] = item
        else:
            if matching_index is not None:
                raise MisconfigurationException(
                    f"Function with name: {name} and metadata: {metadata} is already present within {self}."
                    " HINT: Use `override=True`."
                )
            self.functions.append(item)

    def _find_matching_index(self, item: _REGISTERED_FUNCTION) -> Optional[int]:
        for idx, fn in enumerate(self.functions):
            if all(fn[k] == item[k] for k in ("fn", "name", "metadata")):
                return idx

    def __call__(
        self,
        fn: Optional[Callable[..., Any]] = None,
        name: Optional[str] = None,
        override: bool = False,
        providers: Optional[Union[Provider, List[Provider]]] = None,
        **metadata,
    ) -> Callable:
        """This function is used to register new functions to the registry along their metadata.

        Functions can be filtered using metadata using the ``get`` function.
        """
        if providers is not None:
            metadata["providers"] = providers

        if fn is not None:
            self._register_function(fn=fn, name=name, override=override, metadata=metadata)
            return fn

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):
            raise TypeError(f"`name` must be a str, found {name}")

        def _register(cls):
            self._register_function(fn=cls, name=name, override=override, metadata=metadata)
            return cls

        return _register

    def available_keys(self) -> List[str]:
        return sorted(v["name"] for v in self.functions)
