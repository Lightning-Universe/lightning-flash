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

from flash.core.data.utils import download_data
from flash.core.utilities.apply_func import get_callable_dict, get_callable_name

# ======== Mock functions ========


class A:
    def __call__(self, x):
        return True


def b():
    return True


# ==============================


def test_get_callable_name():
    assert get_callable_name(A()) == "a"
    assert get_callable_name(b) == "b"
    assert get_callable_name(lambda: True) == "<lambda>"


def test_get_callable_dict():
    d = get_callable_dict(A())
    assert type(d["a"]) is A

    d = get_callable_dict([A(), b])
    assert type(d["a"]) is A
    assert d["b"] == b

    d = get_callable_dict({"one": A(), "two": b})
    assert type(d["one"]) is A
    assert d["two"] == b


def test_download_data(tmpdir):
    path = os.path.join(tmpdir, "data")
    download_data("https://pl-flash-data.s3.amazonaws.com/titanic.zip", path)
    assert "titanic" in set(os.listdir(path))
    assert "titanic.zip" in set(os.listdir(path))
