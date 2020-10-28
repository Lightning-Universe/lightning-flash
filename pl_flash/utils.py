from typing import Callable, Union, Mapping, Sequence


def get_callable_name(fn: Callable) -> str:
    return getattr(fn, "__name__", fn.__class__.__name__).lower()


def get_callable_dict(fn: Union[Callable, Mapping, Sequence]) -> dict:
    if isinstance(fn, Mapping):
        return fn
    elif isinstance(fn, Sequence):
        return {get_callable_name(f): f for f in fn}
    elif callable(fn):
        return {get_callable_name(fn): fn}
