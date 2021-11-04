import numpy as np
import pytest
import torch

from flash.core.data.io.input import InputDataKeys
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, _IMAGE_AVAILABLE
from flash.image.detection.output import FiftyOneDetectionLabels


@pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed.")
@pytest.mark.skipif(not _FIFTYONE_AVAILABLE, reason="fiftyone is not installed for testing")
class TestFiftyOneDetectionLabels:
    @staticmethod
    def test_smoke():
        serial = FiftyOneDetectionLabels()
        assert serial is not None

    @staticmethod
    def test_serialize_fiftyone():
        labels = ["class_1", "class_2", "class_3"]
        serial = FiftyOneDetectionLabels()
        filepath_serial = FiftyOneDetectionLabels(return_filepath=True)
        threshold_serial = FiftyOneDetectionLabels(threshold=0.9)
        labels_serial = FiftyOneDetectionLabels(labels=labels)

        sample = {
            InputDataKeys.PREDS: {
                "bboxes": [
                    {
                        "xmin": torch.tensor(20),
                        "ymin": torch.tensor(30),
                        "width": torch.tensor(20),
                        "height": torch.tensor(20),
                    }
                ],
                "labels": [torch.tensor(0)],
                "scores": [torch.tensor(0.5)],
            },
            InputDataKeys.METADATA: {
                "filepath": "something",
                "size": (100, 100),
            },
        }

        detections = serial.transform(sample)
        assert len(detections.detections) == 1
        np.testing.assert_array_almost_equal(detections.detections[0].bounding_box, [0.2, 0.3, 0.2, 0.2])
        assert detections.detections[0].confidence == 0.5
        assert detections.detections[0].label == "0"

        detections = filepath_serial.transform(sample)
        assert len(detections["predictions"].detections) == 1
        np.testing.assert_array_almost_equal(detections["predictions"].detections[0].bounding_box, [0.2, 0.3, 0.2, 0.2])
        assert detections["predictions"].detections[0].confidence == 0.5
        assert detections["filepath"] == "something"

        detections = threshold_serial.transform(sample)
        assert len(detections.detections) == 0

        detections = labels_serial.transform(sample)
        assert len(detections.detections) == 1
        np.testing.assert_array_almost_equal(detections.detections[0].bounding_box, [0.2, 0.3, 0.2, 0.2])
        assert detections.detections[0].confidence == 0.5
        assert detections.detections[0].label == "class_1"
