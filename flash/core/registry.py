import hashlib
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from types import FunctionType
from typing import Any, Callable, Dict, List, Optional, Set, Union

from pytorch_lightning.utilities.exceptions import MisconfigurationException


class FlashRegistry:
    """
    This class is used to register function or partial to a registry:

    Example::

        backbones = FlashRegistry("backbones")

        @backbones.register_function()
        def my_model(nc_input=5, nc_output=6):
            return nn.Linear(nc_input, nc_output), nc_input, nc_output

        mlp, nc_input, nc_output = backbones("my_model")(nc_output=7)

        backbones.register_function(my_model, name="cho")
        assert backbones("cho")

    """

    def __init__(self, registry_name: str, verbose: bool = False) -> None:
        self._registry_name = registry_name
        self._registered_functions: List[Dict[str, Any]] = []
        self._verbose = verbose

    def __len__(self) -> int:
        return len(self._registered_functions)

    def __contains__(self, key) -> bool:
        return any(key == e["name"] for e in self._registered_functions)

    def __repr__(self) -> str:
        format_str = self.__class__.__name__ + \
            f'(name={self._registry_name}, ' \
            f'registered_items={self._registered_functions})'
        return format_str

    @property
    def name(self) -> str:
        return self._registry_name

    @property
    def registered_funcs(self) -> Dict[str, Any]:
        return self._registered_functions

    def validate_matches(self, key: str, matches: Dict, with_metadata: bool, key_in: bool = False):
        if len(matches) == 1:
            registered_function = matches[0]
            if with_metadata:
                return registered_function
            return registered_function["fn"]
        elif len(matches) == 0:
            if key_in:
                raise MisconfigurationException(
                    f"Found {len(matches)} matches within {matches}. Add more metadata to filter them out."
                )
            raise MisconfigurationException(f"Key: {key} is not in {self.__repr__()}")

    def get(self,
            key: str,
            with_metadata: bool = False,
            strict: bool = True,
            **metadata) -> Union[Callable, Dict[str, Any], List[Dict[str, Any]], List[Callable]]:
        """
        This function is used to gather matches from the registry:

        Args:
            key: Name of the registered function.
            with_metadata: Whether to return associated metadata used during registration.
            strict: Whether to return all matches if higher than 1.
            metadata: All filtering metadata used for the registry.

        """
        matches = [e for e in self._registered_functions if key == e["name"]]
        key_in = False
        if len(matches) > 1:
            key_in = True
            matches = self._filter_matches_on_metadata(matches, with_metadata=with_metadata, metadata=metadata)
            if not strict:
                return [e if with_metadata else e["fn"] for e in matches]
            if len(matches) > 1:
                raise MisconfigurationException(
                    f"Found {len(matches)} matches within {matches}. Add more metadata to filter them out."
                )
        return self.validate_matches(key, matches, with_metadata, key_in=key_in)

    def _filter_matches_on_metadata(self, matches, with_metadata: bool = False, **metadata) -> List[Dict[str, Any]]:
        _matches = []
        for item in matches:
            if all(self._extract_metadata(item["metadata"], k) == v for k, v in metadata["metadata"].items()):
                _matches.append(item)
        return _matches

    def remove(self, key: str) -> None:
        _registered_functions = []
        for item in self._registered_functions:
            if item["name"] != key:
                _registered_functions.append(item)
        self._registered_functions = _registered_functions

    def _register_function(
        self, fn: Callable, name: Optional[str] = None, override: bool = False, metadata: Dict[str, Any] = None
    ):
        if not isinstance(fn, FunctionType) and not isinstance(fn, partial):
            raise MisconfigurationException("``register_function`` should be used with a function")

        name = name or fn.__name__

        item = {"fn": fn, "name": name, "metadata": metadata}

        matching_index = self._find_matching_index(item)

        if override and matching_index is not None:
            self._registered_functions[matching_index] = item
        else:
            if matching_index is not None:
                raise MisconfigurationException(
                    f"Function with name: {name} and metadata: {metadata} is already present within {self}."
                    "HINT: Use `override=True`."
                )
            self._registered_functions.append(item)

    @staticmethod
    def _extract_metadata(metadata: Dict, key: str) -> Optional[Any]:
        if key in metadata:
            return metadata[key]

    def _find_matching_index(self, item: Dict[str, Any]) -> Optional[int]:
        for idx, _item in enumerate(self._registered_functions):
            if (
                _item["fn"] == item["fn"] and _item["name"] == item["name"]
                and all(self._extract_metadata(_item["metadata"], k) == v for k, v in item["metadata"].items())
            ):
                return idx

    def __call__(
        self,
        fn: Optional[Callable] = None,
        name: Optional[str] = None,
        override: bool = False,
        **metadata
    ) -> Callable:
        """Register a callable"""
        if fn is not None:
            if self._verbose:
                print(f"Registering: {fn.__name__} function with name: {name} and metadata: {metadata}")
            self._register_function(fn=fn, name=name, override=override, metadata=metadata)
            return fn

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):
            raise TypeError(f'name must be a str, but got {type(name)}')

        def _register(cls):
            self._register_function(fn=cls, name=name, override=override, metadata=metadata)
            return cls

        return _register

    def available_keys(self) -> List[str]:
        return sorted([v["name"] for v in self._registered_functions])


IMAGE_CLASSIFIER_BACKBONES = FlashRegistry("backbones")
OBJ_DETECTION_BACKBONES = FlashRegistry("backbones")
