import pytest
import torch

from flash.core.data.data_source import DefaultDataKeys
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE
from flash.image.segmentation.serialization import FiftyOneSegmentationLabels, SegmentationLabels


class TestSemanticSegmentationLabels:

    @staticmethod
    def test_smoke():
        serial = SegmentationLabels()
        assert serial is not None
        assert serial.labels_map is None
        assert serial.visualize is False

    @staticmethod
    def test_exception():
        serial = SegmentationLabels()

        with pytest.raises(Exception):
            sample = torch.zeros(1, 5, 2, 3)
            serial.serialize(sample)

        with pytest.raises(Exception):
            sample = torch.zeros(2, 3)
            serial.serialize(sample)

    @staticmethod
    def test_serialize():
        serial = SegmentationLabels()

        sample = torch.zeros(5, 2, 3)
        sample[1, 1, 2] = 1  # add peak in class 2
        sample[3, 0, 1] = 1  # add peak in class 4

        classes = serial.serialize({DefaultDataKeys.PREDS: sample})
        assert torch.tensor(classes)[1, 2] == 1
        assert torch.tensor(classes)[0, 1] == 3

    @pytest.mark.skipif(not _FIFTYONE_AVAILABLE, reason="fiftyone is not installed for testing")
    @staticmethod
    def test_serialize_fiftyone():
        serial = FiftyOneSegmentationLabels()
        filepath_serial = FiftyOneSegmentationLabels(return_filepath=True)

        preds = torch.zeros(5, 2, 3)
        preds[1, 1, 2] = 1  # add peak in class 2
        preds[3, 0, 1] = 1  # add peak in class 4

        sample = {
            DefaultDataKeys.PREDS: preds,
            DefaultDataKeys.METADATA: {
                "filepath": "something"
            },
        }

        segmentation = serial.serialize(sample)
        assert segmentation.mask[1, 2] == 1
        assert segmentation.mask[0, 1] == 3

        segmentation = filepath_serial.serialize(sample)
        assert segmentation["predictions"].mask[1, 2] == 1
        assert segmentation["predictions"].mask[0, 1] == 3
        assert segmentation["filepath"] == "something"

    # TODO: implement me
    def test_create_random_labels(self):
        pass

    # TODO: implement me
    def test_labels_to_image(self):
        pass
