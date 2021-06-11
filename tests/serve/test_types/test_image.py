from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import torch

from flash.core.serve.types import Image


def test_deserialize_serialize(session_global_datadir):

    with (session_global_datadir / "cat.jpg").open("rb") as f:
        imgstr = base64.b64encode(f.read()).decode("UTF-8")

    image_type = Image()
    ten = image_type.deserialize(imgstr)
    assert isinstance(ten, torch.Tensor)

    raw = image_type.serialize(ten)
    assert isinstance(raw, str)

    reconstructed = image_type.deserialize(raw)
    assert isinstance(reconstructed, torch.Tensor)
    assert np.allclose(ten.shape, reconstructed.shape)
    assert ten.dtype == reconstructed.dtype
