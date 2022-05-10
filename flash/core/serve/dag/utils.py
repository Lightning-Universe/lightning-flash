"""
NOTICE: Some methods in this file have been modified from their original source.
"""

import functools
import re
from operator import methodcaller

from flash.core.utilities.imports import _SERVE_TESTING

# Skip doctests if requirements aren't available
if not _SERVE_TESTING:
    __doctest_skip__ = ["*"]


def funcname(func):
    """Get the name of a function."""
    # functools.partial
    if isinstance(func, functools.partial):
        return funcname(func.func)
    # methodcaller
    if isinstance(func, methodcaller):
        return str(func)[:50]

    module_name = getattr(func, "__module__", None) or ""
    type_name = getattr(type(func), "__name__", None) or ""

    # cytoolz.curry
    if "cytoolz" in module_name and type_name == "curry":
        return func.func_name[:50]
    # numpy.vectorize objects
    if "numpy" in module_name and type_name == "vectorize":
        return ("vectorize_" + funcname(func.pyfunc))[:50]

    # All other callables
    try:
        name = func.__name__
        if name == "<lambda>":
            return "lambda"
        return name[:50]
    except AttributeError:
        return str(func)[:50]


# Defining `key_split` (used by key renamers in `fuse`) in utils.py
# results in messy circular imports, so define it here instead.
hex_pattern = re.compile("[a-f]+")


def key_split(s):
    """
    >>> key_split('x')
    'x'
    >>> key_split('x-1')
    'x'
    >>> key_split('x-1-2-3')
    'x'
    >>> key_split(('x-2', 1))
    'x'
    >>> key_split("('x-2', 1)")
    'x'
    >>> key_split('hello-world-1')
    'hello-world'
    >>> key_split(b'hello-world-1')
    'hello-world'
    >>> key_split('ae05086432ca935f6eba409a8ecd4896')
    'data'
    >>> key_split('<module.submodule.myclass object at 0xdaf372')
    'myclass'
    >>> key_split(None)
    'Other'
    >>> key_split('x-abcdefab')  # ignores hex
    'x'
    >>> key_split('_(x)')  # strips unpleasant characters
    'x'
    """
    if type(s) is bytes:
        s = s.decode()
    if type(s) is tuple:
        s = s[0]
    try:
        words = s.split("-")
        if not words[0][0].isalpha():
            result = words[0].strip("_'()\"")
        else:
            result = words[0]
        for word in words[1:]:
            if word.isalpha() and not (len(word) == 8 and hex_pattern.match(word) is not None):
                result += "-" + word
            else:
                break
        if len(result) == 32 and re.match(r"[a-f0-9]{32}", result):
            return "data"
        if result[0] == "<":
            result = result.strip("<>").split()[0].split(".")[-1]
        return result
    except Exception:
        return "Other"


def apply(func, args, kwargs=None):
    if kwargs:
        return func(*args, **kwargs)
    return func(*args)


def partial_by_order(*args, **kwargs):
    """
    >>> from operator import add, truediv
    >>> partial_by_order(5, function=add, other=[(1, 10)])
    15
    >>> partial_by_order(10, function=truediv, other=[(1, 5)])
    2.0
    >>> partial_by_order(10, function=truediv, other=[(0, 5)])
    0.5
    """
    function = kwargs.pop("function")
    other = kwargs.pop("other")
    args2 = list(args)
    for i, arg in other:
        args2.insert(i, arg)
    return function(*args2, **kwargs)
