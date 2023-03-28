import operator
from functools import partial

import numpy as np
import pytest

from flash.core.serve.dag.utils import funcname, partial_by_order
from flash.core.utilities.imports import _CYTOOLZ_AVAILABLE, _TOPIC_SERVE_AVAILABLE

if _CYTOOLZ_AVAILABLE:
    from cytoolz import curry


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="Not testing serve.")
def test_funcname_long():
    def a_long_function_name_11111111111111111111111111111111111111111111111():
        pass

    result = funcname(a_long_function_name_11111111111111111111111111111111111111111111111)
    assert "a_long_function_name" in result
    assert len(result) < 60


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="Not testing serve.")
def test_funcname_cytoolz():
    @curry
    def foo(a, b, c):
        pass

    assert funcname(foo) == "foo"
    assert funcname(foo(1)) == "foo"

    def bar(a, b):
        return a + b

    c_bar = curry(bar, 1)
    assert funcname(c_bar) == "bar"


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="Not testing serve.")
def test_partial_by_order():
    assert partial_by_order(5, function=operator.add, other=[(1, 20)]) == 25


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="Not testing serve.")
def test_funcname():
    assert funcname(np.floor_divide) == "floor_divide"
    assert funcname(partial(bool)) == "bool"
    assert funcname(operator.methodcaller("__getitem__")) == "operator.methodcaller('__getitem__')"
    assert funcname(lambda x: x) == "lambda"


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="Not testing serve.")
def test_numpy_vectorize_funcname():
    def myfunc(a, b):
        """Return a-b if a>b, otherwise return a+b."""
        if a > b:
            return a - b
        return a + b

    vfunc = np.vectorize(myfunc)
    assert funcname(vfunc) == "vectorize_myfunc"
