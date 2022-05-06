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
from typing import Callable, Type, Union

from pytorch_lightning.utilities import rank_zero_warn

from flash.core.utilities.imports import _CORE_TESTING

# Skip doctests if requirements aren't available
if not _CORE_TESTING:
    __doctest_skip__ = ["beta"]


@functools.lru_cache()  # Trick to only warn once for each message
def _raise_beta_warning(message: str, stacklevel: int = 6):
    rank_zero_warn(
        f"{message} The API and functionality may change without warning in future releases. "
        "More details: https://lightning-flash.readthedocs.io/en/latest/stability.html",
        stacklevel=stacklevel,
        category=UserWarning,
    )


def beta(message: str = "This feature is currently in Beta."):
    """The beta decorator is used to indicate that a particular feature is in Beta. A callable or type that has
    been marked as beta will give a ``UserWarning`` when it is called or instantiated. This designation should be
    used following the description given in :ref:`stability`.

    Args:
        message: The message to include in the warning.

    Examples
    ________

    .. testsetup::

        >>> import pytest

    .. doctest::

        >>> from flash.core.utilities.stability import beta
        >>> @beta()
        ... class MyBetaFeature:
        ...     pass
        ...
        >>> with pytest.warns(UserWarning, match="This feature is currently in Beta."):
        ...     MyBetaFeature()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        ...
        <...>
        >>> @beta("This feature is currently in Beta with a custom message.")
        ... class MyBetaFeatureWithCustomMessage:
        ...     pass
        ...
        >>> with pytest.warns(UserWarning, match="This feature is currently in Beta with a custom message."):
        ...     MyBetaFeatureWithCustomMessage()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        ...
        <...>
    """

    def decorator(callable: Union[Callable, Type]):
        if inspect.isclass(callable):
            callable.__init__ = decorator(callable.__init__)
            return callable

        @functools.wraps(callable)
        def wrapper(*args, **kwargs):
            _raise_beta_warning(message)
            return callable(*args, **kwargs)

        return wrapper

    return decorator
