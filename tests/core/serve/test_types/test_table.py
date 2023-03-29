import pytest
import torch

from flash.core.serve.types import Table
from flash.core.utilities.imports import _TOPIC_SERVE_AVAILABLE

data = torch.tensor([[0.00632, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2, 4.09, 1.0, 296.0, 15.3, 396.9, 4.98]])
feature_names = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
]


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="Not testing serve.")
def test_serialize_success():
    table = Table(column_names=feature_names)
    sample = data
    dict_data = table.serialize(sample)
    for d1, d2 in zip(sample.squeeze(), dict_data.values()):
        assert d2 == {0: d1.item()}


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="Not testing serve.")
def test_serialize_wrong_shape():
    table = Table(column_names=feature_names)
    sample = data.squeeze()
    with pytest.raises(ValueError):
        # Expected axis has 1 elements, new values have 13 elements
        table.serialize(sample)

    sample = data.unsqueeze(0)
    with pytest.raises(ValueError):
        # Must pass 2-d input. shape=(1, 1, 13)
        table.serialize(sample)

    sample = data[:, 1:]
    with pytest.raises(ValueError):
        # length mismatch
        table.serialize(sample)


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="Not testing serve.")
def test_serialize_without_column_names():
    with pytest.raises(TypeError):
        Table()
    table = Table(feature_names)
    sample = data
    dict_data = table.serialize(sample)
    assert list(dict_data.keys()) == feature_names


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="Not testing serve.")
def test_deserialize():
    arr = torch.tensor([100, 200]).view(1, 2)
    table = Table(column_names=["t1", "t2"])
    assert table.deserialize({"t1": {0: 100}, "t2": {0: 200}}).dtype == torch.int64
    assert table.deserialize({"t1": {0: 100}, "t2": {0: 200.0}}).dtype == torch.float64
    assert torch.allclose(arr, table.deserialize({"t1": {0: 100}, "t2": {0: 200}}))
    with pytest.raises(RuntimeError):
        table.deserialize({"title1": {0: 100}, "title2": {0: 200}})
    assert torch.allclose(
        table.deserialize({"t1": {0: 100.0}, "t2": {1: 200.0}}),
        torch.tensor([[100.0, float("nan")], [float("nan"), 200.0]], dtype=torch.float64),
        equal_nan=True,
    )


@pytest.mark.skipif(not _TOPIC_SERVE_AVAILABLE, reason="Not testing serve.")
def test_deserialize_column_names_failures():
    table = Table(["t1", "t2"])
    with pytest.raises(RuntimeError):
        # different length
        table.deserialize({"title1": {0: 100}})
    with pytest.raises(RuntimeError):
        # different column names but same length
        d = {"tt1": {0: 100}, "tt2": {0: 101}}
        table.deserialize(d)
    with pytest.raises(TypeError):
        # not allowed types
        d = {"t1": {0: 100}, "t2": {0: "dummy string"}}
        table.deserialize(d)
