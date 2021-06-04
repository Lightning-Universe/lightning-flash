import pytest
import torch

from flash.core.utilities.imports import _FIFTYONE_AVAILABLE
from flash.image.detection.serialization import FiftyOneDetectionLabels


@pytest.mark.skipif(not _FIFTYONE_AVAILABLE, reason="fiftyone is not installed for testing")
class TestFiftyOneDetectionLabels:

    def test_smoke(self):
        serial = FiftyOneDetectionLabels()
        assert serial is not None

    def test_serialize_fiftyone(self):
        serial = FiftyOneDetectionLabels()

        sample = [{
            "boxes": [torch.tensor(20), torch.tensor(30), torch.tensor(40), torch.tensor(50)],
            "labels": torch.tensor(0),
            "scores": torch.tensor(0.5),
        }]

        detections = serial.serialize(sample)
        assert len(detections.detections) == 1
        assert detections.detections[0].bounding_box == [20, 30, 20, 20]
