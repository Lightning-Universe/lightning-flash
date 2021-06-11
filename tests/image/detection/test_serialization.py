import pytest
import torch

from flash.core.data.data_source import DefaultDataKeys
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE
from flash.image.detection.serialization import FiftyOneDetectionLabels


@pytest.mark.skipif(not _FIFTYONE_AVAILABLE, reason="fiftyone is not installed for testing")
class TestFiftyOneDetectionLabels:

    def test_smoke(self):
        serial = FiftyOneDetectionLabels()
        assert serial is not None

    def test_serialize_fiftyone(self):
        serial = FiftyOneDetectionLabels()

        sample = {
            DefaultDataKeys.PREDS: [
                {
                    "boxes": [torch.tensor(20), torch.tensor(30),
                              torch.tensor(40), torch.tensor(50)],
                    "labels": torch.tensor(0),
                    "scores": torch.tensor(0.5),
                },
            ],
            DefaultDataKeys.METADATA: {
                "filepath": "something",
                "size": (100, 100),
            },
        }

        detections = serial.serialize(sample)
        assert len(detections.detections) == 1
        assert detections.detections[0].bounding_box == [0.2, 0.3, 0.2, 0.2]
