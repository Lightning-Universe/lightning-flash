import pytest
import torch

from flash.core.serve.types import Label, Repeated
from flash.core.utilities.imports import _SERVE_TESTING


@pytest.mark.skipif(not _SERVE_TESTING)
def test_repeated_deserialize():
    repeated = Repeated(dtype=Label(classes=["classA", "classB"]))
    res = repeated.deserialize(*({"label": "classA"}, {"label": "classA"}, {"label": "classB"}))
    assert res == (torch.tensor(0), torch.tensor(0), torch.tensor(1))


@pytest.mark.skipif(not _SERVE_TESTING)
def test_repeated_serialize(session_global_datadir):
    repeated = Repeated(dtype=Label(path=str(session_global_datadir / "imagenet_labels.txt")))
    assert repeated.deserialize(*({"label": "chickadee"}, {"label": "stingray"})) == (
        torch.tensor(19),
        torch.tensor(6),
    )
    assert repeated.serialize((torch.tensor(19), torch.tensor(6))) == ("chickadee", "stingray")
    assert repeated.serialize(torch.tensor([19, 6])) == ("chickadee", "stingray")


@pytest.mark.skipif(not _SERVE_TESTING)
def test_repeated_max_len():
    repeated = Repeated(dtype=Label(classes=["classA", "classB"]), max_len=2)

    with pytest.raises(ValueError):
        repeated.deserialize(*({"label": "classA"}, {"label": "classA"}, {"label": "classB"}))
    assert repeated.deserialize(*({"label": "classA"}, {"label": "classB"})) == (
        torch.tensor(0),
        torch.tensor(1),
    )
    with pytest.raises(ValueError):
        repeated.serialize((torch.tensor(0), torch.tensor(0), torch.tensor(1)))
    assert repeated.serialize((torch.tensor(1), torch.tensor(0))) == ("classB", "classA")

    # max_len < 1
    with pytest.raises(ValueError):
        Repeated(dtype=Label(classes=["classA", "classB"]), max_len=0)
    assert Repeated(dtype=Label(classes=["classA", "classB"]), max_len=1) is not None

    # type(max_len) is not int
    with pytest.raises(TypeError):
        Repeated(dtype=Label(classes=["classA", "classB"]), max_len=str)


@pytest.mark.skipif(not _SERVE_TESTING)
def test_repeated_non_serve_dtype():
    class NonServeDtype:
        pass

    with pytest.raises(TypeError):
        Repeated(NonServeDtype())


@pytest.mark.skipif(not _SERVE_TESTING)
def test_not_allow_nested_repeated():
    with pytest.raises(TypeError):
        Repeated(dtype=Repeated())
