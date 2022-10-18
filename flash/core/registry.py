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
import inspect
import itertools
from typing import Any, Callable, Dict, List, Optional, Union

from pytorch_lightning.utilities import rank_zero_info

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

    def build_wrapper(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            rank_zero_info(message)
            return func(*args, **kwargs)

        return wrapper

    wrapper = build_wrapper(func)

    if inspect.isclass(func):
        callables = [f for f in dir(func) if callable(getattr(func, f)) and not f.startswith("_")]
        for c in callables:
            setattr(wrapper, c, build_wrapper(getattr(func, c)))

    return wrapper


class FlashRegistry:
    """This class is used to register function or :class:`functools.partial` class to a registry."""

    def __init__(self, name: str, verbose: bool = False) -> None:
        self.name = name
        self.functions: List[_REGISTERED_FUNCTION] = []
        self._verbose = verbose

    def __add__(self, other):
        registries = []
        if isinstance(self, ConcatRegistry):
            registries += self.registries
        else:
            registries += [self]

        if isinstance(other, ConcatRegistry):
            registries = other.registries + tuple(registries)
        else:
            registries = [other] + registries

        return ConcatRegistry(*registries)

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
            raise KeyError(f"Key: {key} is not in {type(self).__name__}. Available keys: {self.available_keys()}")

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
            raise TypeError(f"You can only register a callable, found: {fn}")

        if name is None:
            if hasattr(fn, "func"):
                name = fn.func.__name__
            else:
                name = fn.__name__

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
                raise ValueError(
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


class ExternalRegistry(FlashRegistry):
    """The ``ExternalRegistry`` is a ``FlashRegistry`` that can point to an external provider via a getter
    function.

    Args:
        getter: A function whose first argument is a key that can optionally take additional args and kwargs.
        providers: The provider(/s) of entries in this registry.
    """

    # Prevent users from trying to remove or register items
    remove = None
    _register_function = None

    def __init__(
        self,
        getter: Callable,
        name: str,
        providers: Optional[Union[Provider, List[Provider]]] = None,
        verbose: bool = False,
        **metadata,
    ):
        super().__init__(name, verbose=verbose)

        self.getter = getter
        self.providers = providers if providers is None or isinstance(providers, list) else [providers]
        self.metadata = metadata

    def __contains__(self, item):
        """Contains is always ``True`` for an ``ExternalRegistry`` as we can't know whether the getter will fail
        without executing it."""
        return True

    def get(
        self,
        key: str,
        with_metadata: bool = False,
        strict: bool = True,
        **metadata,
    ) -> Union[Callable, _REGISTERED_FUNCTION, List[_REGISTERED_FUNCTION], List[Callable]]:
        """Returns a partial of the getter with the first argument as the given key and wrapped to print the
        providers."""
        fn = functools.partial(self.getter, key)
        if self.providers is not None:
            fn = print_provider_info(key, self.providers, fn)

        if not with_metadata:
            return fn
        return {"fn": fn, "metadata": self.metadata}

    def available_keys(self) -> List[str]:
        """Since we don't know the available keys, just give a generic message."""
        if self.providers is not None:
            return [f"Anything available from: {', '.join(str(provider) for provider in self.providers)}"]
        return []


class ConcatRegistry(FlashRegistry):
    """The ``ConcatRegistry`` can be used to concatenate multiple registries of different types together."""

    def __init__(self, *registries: FlashRegistry):
        super().__init__(
            ",".join(
                {
                    registry.name
                    for registry in sorted(registries, key=lambda r: 1 if isinstance(r, ExternalRegistry) else 0)
                }
            ),
            verbose=any(registry._verbose for registry in registries),
        )

        self.registries = registries

    def __len__(self) -> int:
        return sum(len(registry) for registry in self.registries)

    def __contains__(self, key) -> bool:
        return any(key in registry for registry in self.registries)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(registries={self.registries})"

    def get(
        self,
        key: str,
        with_metadata: bool = False,
        strict: bool = True,
        **metadata,
    ) -> Union[Callable, _REGISTERED_FUNCTION, List[_REGISTERED_FUNCTION], List[Callable]]:
        matches = []
        external_matches = []

        for registry in self.registries:
            if key in registry:
                result = registry.get(key, with_metadata=with_metadata, strict=strict, **metadata)
                if not isinstance(result, list):
                    result = [result]

                if isinstance(registry, ExternalRegistry):
                    external_matches += result
                else:
                    matches += result

        if not strict:
            return matches + external_matches

        if len(matches) > 0:
            return matches[0]

        if len(external_matches) == 1:
            return external_matches[0]

        if len(matches) == 0 and len(external_matches) == 0:
            raise KeyError("No matches found in registry.")
        raise KeyError("Multiple matches from external registries, a strict lookup is not possible.")

    def remove(self, key: str) -> None:
        for registry in self.registries:
            if key in registry and getattr(registry, "remove", None) is not None:
                registry.remove(key)

    def _register_function(
        self,
        fn: Callable,
        name: Optional[str] = None,
        override: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Register in the first available registry."""
        for registry in self.registries:
            if getattr(registry, "_register_function", None) is not None:
                return registry._register_function(fn, name=name, override=override, metadata=metadata)

    def available_keys(self) -> List[str]:
        return list(itertools.chain.from_iterable(registry.available_keys() for registry in self.registries))
