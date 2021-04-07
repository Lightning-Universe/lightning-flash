from collections import defaultdict
from functools import partial
from types import FunctionType
from typing import Callable, Dict, Mapping, Optional, Union

from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.nn import Module


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
        self._registered_functions: Mapping[str, Callable] = defaultdict()
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
    def registered_funcs(self):
        return self._registered_functions

    def __getitem__(self, key: str) -> Optional[Callable]:
        return self.get(key)

    def get(self, key: str) -> Optional[Callable]:
        if key in self._registered_functions:
            fn = self._registered_functions[key]
            return fn
        else:
            raise MisconfigurationException(f"Key: {key} is not in {self.__repr__()}")

    def _register_function(self, fn: Callable, name: Optional[str] = None, override: bool = False):
        if not isinstance(fn, FunctionType) and not isinstance(fn, partial):
            raise MisconfigurationException("``register_function`` should be used with a function")

        name = name or fn.__name__

        if override:
            self._registered_functions[name] = fn
        else:
            if name in self._registered_functions:
                raise MisconfigurationException(f"Name {name} is already present within {self}")
            self._registered_functions[name] = fn

    def register_function(
        self, fn: Optional[Callable] = None, name: Optional[str] = None, override: bool = False
    ) -> Callable:
        """Register a callable
        """
        if fn is not None:
            if self._verbose:
                print(f"Registering: {fn.__name__} function with {name}")
            self._register_function(fn=fn, name=name, override=override)
            return fn

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):
            raise TypeError(f'name must be a str, but got {type(name)}')

        def _register(cls):
            self._register_function(fn=cls, name=name, override=override)
            return cls

        return _register


BACKBONES_REGISTRY = FlashRegistry("backbones")
