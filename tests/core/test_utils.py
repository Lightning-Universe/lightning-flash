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
