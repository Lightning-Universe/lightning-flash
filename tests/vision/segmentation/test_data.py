from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
from PIL import Image

from flash import Trainer
from flash.vision import SemanticSegmentation, SemanticSegmentationData, SemantincSegmentationPreprocess


def build_checkboard(n, m, k=8):
    x = np.zeros((n, m))
    x[k::k * 2, ::k] = 1
    x[::k * 2, k::k * 2] = 1
    return x


def _rand_image(size: Tuple[int, int]):
    data = build_checkboard(*size).astype(np.uint8)[..., None].repeat(3, -1)
    return Image.fromarray(data)


# usually labels come as rgb images -> need to map to labels
def _rand_labels(size: Tuple[int, int], map_labels: Dict[int, Tuple[int, int, int]] = None):
    data: np.ndarray = np.random.rand(*size, 3) / .5
    if map_labels is not None:
        data_bin = (data.mean(-1) > 0.5)
        for k, v in map_labels.items():
            mask = (data_bin == k)
            data[mask] = v
    return Image.fromarray(data.astype(np.uint8))


def create_random_data(
    image_files: List[str],
    label_files: List[str],
    size: Tuple[int, int],
    map_labels: Optional[Dict[int, Tuple[int, int, int]]] = None,
):
    for img_file in image_files:
        _rand_image(size).save(img_file)

    for label_file in label_files:
        _rand_labels(size, map_labels).save(label_file)


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
            str(tmp_dir / "img1.png"),
            str(tmp_dir / "img2.png"),
            str(tmp_dir / "img3.png"),
        ]

        train_labels = [
            str(tmp_dir / "labels_img1.png"),
            str(tmp_dir / "labels_img2.png"),
            str(tmp_dir / "labels_img3.png"),
        ]

        img_size: Tuple[int, int] = (196, 196)
        create_random_data(train_images, train_labels, img_size)

        # instantiate the data module

        dm = SemanticSegmentationData.from_filepaths(
            train_filepaths=train_images,
            train_labels=train_labels,
            val_filepaths=train_images,
            val_labels=train_labels,
            test_filepaths=train_images,
            test_labels=train_labels,
            batch_size=2,
            num_workers=0,
        )
        assert dm is not None
        assert dm.train_dataloader() is not None
        assert dm.val_dataloader() is not None
        assert dm.test_dataloader() is not None

        # check training data
        data = next(iter(dm.train_dataloader()))
        imgs, labels = data
        assert imgs.shape == (2, 3, 196, 196)
        assert labels.shape == (2, 3, 196, 196)

        # check training data
        data = next(iter(dm.val_dataloader()))
        imgs, labels = data
        assert imgs.shape == (2, 3, 196, 196)
        assert labels.shape == (2, 3, 196, 196)

        # check training data
        data = next(iter(dm.val_dataloader()))
        imgs, labels = data
        assert imgs.shape == (2, 3, 196, 196)
        assert labels.shape == (2, 3, 196, 196)

    def test_map_labels(self, tmpdir):
        tmp_dir = Path(tmpdir)

        # create random dummy data

        train_images = [
            str(tmp_dir / "img1.png"),
            str(tmp_dir / "img2.png"),
            str(tmp_dir / "img3.png"),
        ]

        train_labels = [
            str(tmp_dir / "labels_img1.png"),
            str(tmp_dir / "labels_img2.png"),
            str(tmp_dir / "labels_img3.png"),
        ]

        map_labels: Dict[int, Tuple[int, int, int]] = {
            0: [0, 0, 0],
            1: [255, 255, 255],
        }

        img_size: Tuple[int, int] = (196, 196)
        create_random_data(train_images, train_labels, img_size, map_labels)

        # instantiate the data module

        dm = SemanticSegmentationData.from_filepaths(
            train_filepaths=train_images,
            train_labels=train_labels,
            val_filepaths=train_images,
            val_labels=train_labels,
            batch_size=2,
            num_workers=0,
            map_labels=map_labels
        )
        assert dm is not None
        assert dm.train_dataloader() is not None

        # disable visualisation for testing
        assert dm.data_fetcher.block_viz_window is True
        dm.set_block_viz_window(False)
        assert dm.data_fetcher.block_viz_window is False

        dm.set_map_labels(map_labels)
        dm.show_train_batch("load_sample")
        dm.show_train_batch("to_tensor_transform")

        # check training data
        data = next(iter(dm.train_dataloader()))
        imgs, labels = data
        assert imgs.shape == (2, 3, 196, 196)
        assert labels.shape == (2, 196, 196)
        assert labels.min().item() == 0
        assert labels.max().item() == 1
        assert labels.dtype == torch.int64

        # now train with `fast_dev_run`
        model = SemanticSegmentation(num_classes=2, backbone="torchvision/fcn_resnet50")
        trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
        trainer.finetune(model, dm, strategy="freeze_unfreeze")
