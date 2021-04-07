import hashlib
from collections import defaultdict
from functools import partial
from types import FunctionType
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Union

from pytorch_lightning.utilities.exceptions import MisconfigurationException


class FlashRegistry(Dict):
    """
    This class is used to register function or partial to a registry:

    Example::

        backbones = FlashRegistry("backbones")

        @backbones.register_function()
        def my_model(nc_input=5, nc_output=6):
            return nn.Linear(nc_input, nc_output), nc_input, nc_output

        mlp, nc_input, nc_output = backbones.get("my_model")(nc_output=7)

        backbones.register_function(my_model, name="cho")
        assert backbones.get("cho")

    """

    def __init__(self, registry_name: str, verbose: bool = False):
        self._registry_name = registry_name
        self._registered_functions: Mapping[str, Dict[str, Any]] = defaultdict()
        self._registered_functions_mapping: Dict[str, str] = {}
        self._verbose = verbose

    def __len__(self):
        return len(self._registered_functions)

    def __contains__(self, key):
        return self._registered_functions.get(key, None)

    def __repr__(self):
        format_str = self.__class__.__name__ + \
            f'(name={self._registry_name}, ' \
            f'registered_items={dict(**self._registered_functions)})'
        return format_str

    @property
    def name(self) -> str:
        return self._registry_name

    @property
    def registered_funcs(self) -> Dict[str, Any]:
        return self._registered_functions

    def __getitem__(self, key: str) -> Callable:
        return self.get(key)

    def validate_matches(self, key: str, matches: Dict, with_metadata: bool, key_in: bool = False):
        if len(matches) == 1:
            registered_function = self._registered_functions[list(matches.keys())[0]]
            if with_metadata:
                return registered_function
            return registered_function["fn"]
        elif len(matches) == 0:
            if key_in:
                raise MisconfigurationException(
                    f"Found {len(matches)} matches within {matches}. Add more metadata to filter them out."
                )
            raise MisconfigurationException(f"Key: {key} is not in {self.__repr__()}")

    def __call__(self,
                 key: str,
                 with_metadata: bool = False,
                 strict: bool = True,
                 **metadata) -> Union[Callable, Dict[str, Any], List[Dict[str, Any]], List[Callable]]:

        return self.get(key, with_metadata=with_metadata, strict=strict, **metadata)

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
        matches = {_hash: name for _hash, name in self._registered_functions_mapping.items() if key == name}
        key_in = False
        if len(matches) > 1:
            key_in = True
            matches = self._filter_matches_on_metadata(matches, with_metadata=with_metadata, metadata=metadata)
            if not strict:
                _matches = []
                for v in matches:
                    match = list(v.values())[0]
                    if with_metadata:
                        match = match["fn"]
                    _matches.append(match)
                return _matches
            if len(matches) > 1:
                raise MisconfigurationException(
                    f"Found {len(matches)} matches within {matches}. Add more metadata to filter them out."
                )
            elif len(matches) == 1:
                matches = matches[0]
        return self.validate_matches(key, matches, with_metadata, key_in=key_in)

    def remove(self, key: str) -> None:
        matches = {hash for hash, _key in self._registered_functions_mapping.items() if key == _key}
        for hash in matches:
            del self._registered_functions_mapping[hash]
            del self._registered_functions[hash]

    def _register_function(
        self, fn: Callable, name: Optional[str] = None, override: bool = False, metadata: Dict[str, Any] = None
    ):
        if not isinstance(fn, FunctionType) and not isinstance(fn, partial):
            raise MisconfigurationException("``register_function`` should be used with a function")

        name = name or fn.__name__

        registered_function = {"fn": fn, "name": name, "metadata": metadata}

        hash_algo = hashlib.sha256()
        hash_algo.update(str(name + str(metadata)).encode('utf-8'))
        hash = hash_algo.hexdigest()

        if override:
            self._registered_functions[hash] = registered_function
        else:
            if hash in self._registered_functions_mapping:
                raise MisconfigurationException(
                    f"Function with name: {name} and metadata: {metadata} is already present within {self}"
                )
            self._registered_functions[hash] = registered_function
            self._registered_functions_mapping.update({hash: name})

    def register_function(
        self,
        fn: Optional[Callable] = None,
        name: Optional[str] = None,
        override: bool = False,
        **metadata
    ) -> Callable:
        """Register a callable
        """
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

    def _filter_matches_on_metadata(self, matches, with_metadata: bool = False, **metadata) -> List[Dict[str, Any]]:

        def _extract_metadata(metadata: Dict, key: str) -> Optional[Any]:
            if key in metadata:
                return metadata[key]

        _matches = []
        for hash in matches.keys():
            registered_function = self._registered_functions[hash]
            _metadata = registered_function["metadata"]
            if all(_extract_metadata(_metadata, k) == v for k, v in metadata["metadata"].items()):
                _matches.append({hash: registered_function})
        return _matches


BACKBONES_REGISTRY = FlashRegistry("backbones")
