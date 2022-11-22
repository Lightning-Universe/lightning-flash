import base64

import numpy as np
import pytest
from torch import Tensor

from flash.core.serve.types import Image
from flash.core.utilities.imports import _PIL_AVAILABLE, _SERVE_TESTING


@pytest.mark.skipif(not _SERVE_TESTING, reason="Not testing serve.")
@pytest.mark.skipif(not _PIL_AVAILABLE, reason="library PIL is not installed.")
def test_deserialize_serialize(session_global_datadir):
    with (session_global_datadir / "cat.jpg").open("rb") as f:
        imgstr = base64.b64encode(f.read()).decode("UTF-8")

    image_type = Image()
    ten = image_type.deserialize(imgstr)
    assert isinstance(ten, Tensor)

    raw = image_type.serialize(ten)
    assert isinstance(raw, str)

    reconstructed = image_type.deserialize(raw)
    assert isinstance(reconstructed, Tensor)
    assert np.allclose(ten.shape, reconstructed.shape)
    assert ten.dtype == reconstructed.dtype
