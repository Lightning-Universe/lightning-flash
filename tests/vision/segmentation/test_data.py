from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from PIL import Image

from flash.vision import SemanticSegmentationData, SemantincSegmentationPreprocess


def _rand_image(size: Tuple[int, int]):
    data: np.ndarray = np.random.randint(0, 255, (*size, 3), dtype="uint8")
    return Image.fromarray(data)


# usually labels come as rgb images -> need to map to labels
def _rand_labels(size: Tuple[int, int]):
    data: np.ndarray = np.random.randint(0, 255, (*size, 3), dtype="uint8")
    return Image.fromarray(data)


def create_random_data(image_files: List[str], label_files: List[str], size: Tuple[int, int]) -> Image.Image:
    for img_file in image_files:
        _rand_image(size).save(img_file)

    for label_file in label_files:
        _rand_labels(size).save(img_file)


class TestSemanticSegmentationPreprocess:

    @pytest.mark.xfail(reaspn="parameters are marked as optional but it returns Misconficg error.")
    def test_smoke(self):
        prep = SemantincSegmentationPreprocess()
        assert prep is not None


class TestSemanticSegmentationData:

    def test_smoke(self):
        dm = SemanticSegmentationData()
        assert dm is not None

    def test_from_filepaths(self, tmpdir):
        tmp_dir = Path(tmpdir)

        # create random dummy data

        train_images = [
            tmp_dir / "img1.png",
            tmp_dir / "img2.png",
            tmp_dir / "img3.png",
        ]

        train_labels = [
            tmp_dir / "labels_img1.png",
            tmp_dir / "labels_img2.png",
            tmp_dir / "labels_img3.png",
        ]

        img_size: Tuple[int, int] = (192, 192)
        create_random_data(train_images, train_labels, img_size)

        # instantiate the data module

        dm = SemanticSegmentationData.from_filepaths(
            train_filepaths=train_images,
            train_labels=train_labels,
            batch_size=2,
            num_workers=0,
        )
        assert dm is not None
        # assert dm.train_dataloader() is not None
