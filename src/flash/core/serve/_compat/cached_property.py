"""Backport of python 3.8 functools.cached_property.

cached_property() - computed once per instance, cached as attribute

credits: https://github.com/penguinolog/backports.cached_property

"""

__all__ = ("cached_property",)

# Standard Library
from functools import cached_property  # pylint: disable=no-name-in-module

# Standard Library
