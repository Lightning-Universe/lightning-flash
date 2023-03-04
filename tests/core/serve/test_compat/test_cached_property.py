"""Tests for cached_property.
1. Tests ported from python standard library.
2. Validation for python 3.8+ to use standard library.

credits: https://github.com/penguinolog/backports.cached_property
"""

# Standard Library
import concurrent.futures
import sys
import threading

import pytest

# Package Implementation
from flash.core.serve._compat.cached_property import cached_property
from flash.core.utilities.imports import _SERVE_TESTING


class CachedCostItem:
    """Simple cached property with classvar."""

    _cost = 1

    def __init__(self):
        self.lock = threading.RLock()

    @cached_property
    def cost(self):
        """The cost of the item."""
        with self.lock:
            self._cost += 1
        return self._cost


class OptionallyCachedCostItem:
    """Cached property with non-cached getter available."""

    _cost = 1

    def get_cost(self):
        """The cost of the item."""
        self._cost += 1
        return self._cost

    cached_cost = cached_property(get_cost)


class CachedCostItemWait:
    """Cached property with waiting for event."""

    def __init__(self, event):
        self._cost = 1
        self.lock = threading.RLock()
        self.event = event

    @cached_property
    def cost(self):
        """The cost of the item."""
        self.event.wait(1)
        with self.lock:
            self._cost += 1
        return self._cost


class CachedCostItemWithSlots:
    """Slots implemented without __dict__."""

    __slots__ = ["_cost"]

    def __init__(self):
        self._cost = 1

    @cached_property
    def cost(self):
        """The cost of the item."""
        raise RuntimeError("never called, slots not supported")


# noinspection PyStatementEffect
@pytest.mark.skipif(not _SERVE_TESTING, reason="Not testing serve.")
class TestCachedProperty:
    @staticmethod
    def test_cached():
        item = CachedCostItem()
        assert item.cost == 2
        assert item.cost == 2  # not 3

    @staticmethod
    def test_cached_attribute_name_differs_from_func_name():
        item = OptionallyCachedCostItem()
        assert item.get_cost() == 2
        assert item.cached_cost == 3
        assert item.get_cost() == 4
        assert item.cached_cost == 3

    @staticmethod
    def test_threaded():
        go = threading.Event()
        item = CachedCostItemWait(go)

        num_threads = 3

        orig_si = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            tpr = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix="test")
            futures = [tpr.submit(lambda: item.cost) for _ in range(num_threads)]
            _, not_done = concurrent.futures.wait(futures)
            # "Threads not stopped"
            assert len(not_done) == 0
        finally:
            sys.setswitchinterval(orig_si)

        assert item.cost == 2

    @staticmethod
    def test_object_with_slots():
        item = CachedCostItemWithSlots()
        with pytest.raises(
            TypeError,
            match="No '__dict__' attribute on 'CachedCostItemWithSlots' instance to cache 'cost' property.",
        ):
            item.cost

    @staticmethod
    def test_immutable_dict():
        class MyMeta(type):
            """Test metaclass."""

            @cached_property
            def prop(self):
                """Property impossible to cache standard way."""
                return True

        class MyClass(metaclass=MyMeta):
            """Test class."""

        with pytest.raises(
            TypeError,
            match="The '__dict__' attribute on 'MyMeta' instance does not support",
        ):
            MyClass.prop

    @staticmethod
    def test_reuse_different_names():
        """Disallow this case because decorated function a would not be cached."""
        with pytest.raises(RuntimeError):
            # noinspection PyUnusedLocal
            class ReusedCachedProperty:  # NOSONAR
                """Test class."""

                # noinspection PyPropertyDefinition
                @cached_property
                def a(self):  # NOSONAR
                    """Test getter."""

                b = a

    @staticmethod
    def test_reuse_same_name():
        """Reusing a cached_property on different classes under the same name is OK."""
        counter = 0

        @cached_property
        def _cp(_self):
            nonlocal counter
            counter += 1
            return counter

        class A:  # NOSONAR
            """Test class 1."""

            cp = _cp

        class B:  # NOSONAR
            """Test class 2."""

            cp = _cp

        a = A()
        b = B()

        assert a.cp == 1
        assert b.cp == 2
        assert a.cp == 1

    @staticmethod
    def test_set_name_not_called():
        cp = cached_property(lambda s: None)

        class Foo:
            """Test class."""

        Foo.cp = cp

        with pytest.raises(
            TypeError,
            match="Cannot use cached_property instance without calling __set_name__ on it.",
        ):
            # noinspection PyStatementEffect,PyUnresolvedReferences
            Foo().cp

    @staticmethod
    def test_access_from_class():
        assert isinstance(CachedCostItem.cost, cached_property)

    @staticmethod
    def test_doc():
        assert CachedCostItem.cost.__doc__ == "The cost of the item."


@pytest.mark.skipif(not _SERVE_TESTING, reason="Not testing serve.")
@pytest.mark.skipif(sys.version_info < (3, 8), reason="Validate, that python 3.8 uses standard implementation")
class TestPy38Plus:
    @staticmethod
    def test_is():
        import functools

        # "Python 3.8+ should use standard implementation.")
        assert cached_property is functools.cached_property
