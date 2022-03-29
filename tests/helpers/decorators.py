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
import multiprocessing as mp
import os

import pytest
from dill import dumps, loads


def forked(callable):
    # PyTest forked not available in Windows
    if os.name == "nt":
        return callable
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
    return pytest.mark.forked(callable)


class pickleable_target:
    def __init__(self, target):
        self.target = target

    def __call__(self, *args, **kwargs):
        return self.target(*args, **kwargs)

    def __getstate__(self):
        self.target = dumps(self.target)
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d
        self.target = loads(self.target)


class spawned:
    def __init__(self, target):
        self.target = target
        functools.update_wrapper(self, target)

    def __call__(self, *args, **kwargs):
        context = mp.get_context("spawn")
        target = pickleable_target(self.target)

        p = context.Process(target=target, args=args, kwargs=kwargs)
        p.start()
        p.join()
        assert not p.exitcode
