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
import os

from flash import utils
from flash.core.data import download_data

# ======== Mock functions ========


class A:

    def __call__(self, x):
        return True


def b():
    return True


c = lambda: True  # noqa: E731

# ==============================


def test_get_callable_name():
    assert utils.get_callable_name(A()) == "a"
    assert utils.get_callable_name(b) == "b"
    assert utils.get_callable_name(c) == "<lambda>"


def test_get_callable_dict():
    d = utils.get_callable_dict(A())
    assert type(d["a"]) == A

    d = utils.get_callable_dict([A(), b])
    assert type(d["a"]) == A
    assert d["b"] == b

    d = utils.get_callable_dict({"one": A(), "two": b, "three": c})
    assert type(d["one"]) == A
    assert d["two"] == b
    assert d["three"] == c


def test_download_data(tmpdir):
    path = os.path.join(tmpdir, "data")
    download_data("https://pl-flash-data.s3.amazonaws.com/titanic.zip", path)
    assert set(os.listdir(path)) == {'titanic', 'titanic.zip'}
