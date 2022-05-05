import pytest
import torch

from flash.core.serve.types import BBox
from flash.core.utilities.imports import _SERVE_TESTING


@pytest.mark.skipif(not _SERVE_TESTING)
def test_deserialize():
    bbox = BBox()
    assert torch.allclose(bbox.deserialize((0, 0, 0, 0)), torch.zeros((4,)))
    assert bbox.deserialize((0, 0, 0, 0)).shape == torch.Size([4])
    with pytest.raises(ValueError):
        # only three elements, need four
        bbox.deserialize((0, 1, 2))
    with pytest.raises(ValueError):
        # string in value
        bbox.deserialize(("hai", 1, 2, 3))
    with pytest.raises(TypeError):
        # dictionary
        bbox.deserialize({1: 1, 2: 2, 3: 3, 4: 4})
    with pytest.raises(ValueError):
        # tuple instead of float
        bbox.deserialize(
            (
                (
                    0,
                    0,
                ),
                (0, 0),
                (0, 0),
                (0, 0),
            )
        )


@pytest.mark.skipif(not _SERVE_TESTING)
def test_serialize():
    bbox = BBox()
    assert bbox.serialize(torch.ones(4)) == [1.0, 1.0, 1.0, 1.0]
    assert bbox.serialize(torch.zeros((1, 4))) == [0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        # dimension
        assert bbox.serialize(torch.ones((2, 4)))
    with pytest.raises(TypeError):
        # unsupported type
        bbox.serialize(torch.randn(1, 4, dtype=torch.cfloat))
