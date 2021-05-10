import pytest
import torch

from flash.vision.segmentation.serialization import SegmentationLabels


class TestSemanticSegmentationLabels:

    def test_smoke(self):
        serial = SegmentationLabels()
        assert serial is not None
        assert serial.labels_map is None
        assert serial.visualize is False

    def test_serialize(self):
        serial = SegmentationLabels()

        sample = torch.zeros(5, 2, 3)
        sample[1, 1, 2] = 1  # add peak in class 2
        sample[3, 0, 1] = 1  # add peak in class 4

        classes = serial.serialize(sample)
        assert classes[1, 2] == 1
        assert classes[0, 1] == 3

    # TODO: implement me
    def test_create_random_labels(self):
        pass

    # TODO: implement me
    def test_labels_to_image(self):
        pass
