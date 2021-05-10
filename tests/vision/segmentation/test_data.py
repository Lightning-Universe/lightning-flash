import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch
from PIL import Image

from flash import Trainer
from flash.data.data_source import DefaultDataKeys
from flash.vision import SemanticSegmentation, SemanticSegmentationData, SemanticSegmentationPreprocess


def build_checkboard(n, m, k=8):
    x = np.zeros((n, m))
    x[k::k * 2, ::k] = 1
    x[::k * 2, k::k * 2] = 1
    return x


def _rand_image(size: Tuple[int, int]):
    data = build_checkboard(*size).astype(np.uint8)[..., None].repeat(3, -1)
    return Image.fromarray(data)


# usually labels come as rgb images -> need to map to labels
def _rand_labels(size: Tuple[int, int], num_classes: int):
    data: np.ndarray = np.random.randint(0, num_classes, (*size, 1))
    data = data.repeat(3, axis=-1)
    return Image.fromarray(data.astype(np.uint8))


def create_random_data(image_files: List[str], label_files: List[str], size: Tuple[int, int], num_classes: int):
    for img_file in image_files:
        _rand_image(size).save(img_file)

    for label_file in label_files:
        _rand_labels(size, num_classes).save(label_file)


class TestSemanticSegmentationPreprocess:

    @pytest.mark.xfail(reaspn="parameters are marked as optional but it returns Misconficg error.")
    def test_smoke(self):
        prep = SemanticSegmentationPreprocess()
        assert prep is not None


class TestSemanticSegmentationData:

    def test_smoke(self):
        dm = SemanticSegmentationData()
        assert dm is not None

    def test_from_folders(self, tmpdir):
        tmp_dir = Path(tmpdir)

        # create random dummy data

        os.makedirs(str(tmp_dir / "images"))
        os.makedirs(str(tmp_dir / "targets"))

        images = [
            str(tmp_dir / "images" / "img1.png"),
            str(tmp_dir / "images" / "img2.png"),
            str(tmp_dir / "images" / "img3.png"),
        ]

        targets = [
            str(tmp_dir / "targets" / "img1.png"),
            str(tmp_dir / "targets" / "img2.png"),
            str(tmp_dir / "targets" / "img3.png"),
        ]

        num_classes: int = 2
        img_size: Tuple[int, int] = (196, 196)
        create_random_data(images, targets, img_size, num_classes)

        # instantiate the data module

        dm = SemanticSegmentationData.from_folders(
            train_folder=str(tmp_dir / "images"),
            train_target_folder=str(tmp_dir / "targets"),
            val_folder=str(tmp_dir / "images"),
            val_target_folder=str(tmp_dir / "targets"),
            test_folder=str(tmp_dir / "images"),
            test_target_folder=str(tmp_dir / "targets"),
            batch_size=2,
            num_workers=0,
        )
        assert dm is not None
        assert dm.train_dataloader() is not None
        assert dm.val_dataloader() is not None
        assert dm.test_dataloader() is not None

        # check training data
        data = next(iter(dm.train_dataloader()))
        imgs, labels = data[DefaultDataKeys.INPUT], data[DefaultDataKeys.TARGET]
        assert imgs.shape == (2, 3, 196, 196)
        assert labels.shape == (2, 196, 196)

        # check val data
        data = next(iter(dm.val_dataloader()))
        imgs, labels = data[DefaultDataKeys.INPUT], data[DefaultDataKeys.TARGET]
        assert imgs.shape == (2, 3, 196, 196)
        assert labels.shape == (2, 196, 196)

        # check test data
        data = next(iter(dm.test_dataloader()))
        imgs, labels = data[DefaultDataKeys.INPUT], data[DefaultDataKeys.TARGET]
        assert imgs.shape == (2, 3, 196, 196)
        assert labels.shape == (2, 196, 196)

    def test_from_folders_warning(self, tmpdir):
        tmp_dir = Path(tmpdir)

        # create random dummy data

        os.makedirs(str(tmp_dir / "images"))
        os.makedirs(str(tmp_dir / "targets"))

        images = [
            str(tmp_dir / "images" / "img1.png"),
            str(tmp_dir / "images" / "img3.png"),
        ]

        targets = [
            str(tmp_dir / "targets" / "img1.png"),
            str(tmp_dir / "targets" / "img2.png"),
        ]

        num_classes: int = 2
        img_size: Tuple[int, int] = (196, 196)
        create_random_data(images, targets, img_size, num_classes)

        # instantiate the data module

        with pytest.warns(UserWarning, match="Found inconsistent files"):
            dm = SemanticSegmentationData.from_folders(
                train_folder=str(tmp_dir / "images"),
                train_target_folder=str(tmp_dir / "targets"),
                batch_size=1,
                num_workers=0,
            )
        assert dm is not None
        assert dm.train_dataloader() is not None

        # check training data
        data = next(iter(dm.train_dataloader()))
        imgs, labels = data[DefaultDataKeys.INPUT], data[DefaultDataKeys.TARGET]
        assert imgs.shape == (1, 3, 196, 196)
        assert labels.shape == (1, 196, 196)

    def test_from_files(self, tmpdir):
        tmp_dir = Path(tmpdir)

        # create random dummy data

        images = [
            str(tmp_dir / "img1.png"),
            str(tmp_dir / "img2.png"),
            str(tmp_dir / "img3.png"),
        ]

        targets = [
            str(tmp_dir / "labels_img1.png"),
            str(tmp_dir / "labels_img2.png"),
            str(tmp_dir / "labels_img3.png"),
        ]

        num_classes: int = 2
        img_size: Tuple[int, int] = (196, 196)
        create_random_data(images, targets, img_size, num_classes)

        # instantiate the data module

        dm = SemanticSegmentationData.from_files(
            train_files=images,
            train_targets=targets,
            val_files=images,
            val_targets=targets,
            test_files=images,
            test_targets=targets,
            batch_size=2,
            num_workers=0,
        )
        assert dm is not None
        assert dm.train_dataloader() is not None
        assert dm.val_dataloader() is not None
        assert dm.test_dataloader() is not None

        # check training data
        data = next(iter(dm.train_dataloader()))
        imgs, labels = data[DefaultDataKeys.INPUT], data[DefaultDataKeys.TARGET]
        assert imgs.shape == (2, 3, 196, 196)
        assert labels.shape == (2, 196, 196)

        # check val data
        data = next(iter(dm.val_dataloader()))
        imgs, labels = data[DefaultDataKeys.INPUT], data[DefaultDataKeys.TARGET]
        assert imgs.shape == (2, 3, 196, 196)
        assert labels.shape == (2, 196, 196)

        # check test data
        data = next(iter(dm.test_dataloader()))
        imgs, labels = data[DefaultDataKeys.INPUT], data[DefaultDataKeys.TARGET]
        assert imgs.shape == (2, 3, 196, 196)
        assert labels.shape == (2, 196, 196)

    def test_from_files_warning(self, tmpdir):
        tmp_dir = Path(tmpdir)

        # create random dummy data

        images = [
            str(tmp_dir / "img1.png"),
            str(tmp_dir / "img2.png"),
            str(tmp_dir / "img3.png"),
        ]

        targets = [
            str(tmp_dir / "labels_img1.png"),
            str(tmp_dir / "labels_img2.png"),
            str(tmp_dir / "labels_img3.png"),
        ]

        num_classes: int = 2
        img_size: Tuple[int, int] = (196, 196)
        create_random_data(images, targets, img_size, num_classes)

        # instantiate the data module

        with pytest.warns(UserWarning, match="The number of input files"):
            dm = SemanticSegmentationData.from_files(
                train_files=images,
                train_targets=targets + [str(tmp_dir / "labels_img4.png")],
                batch_size=2,
                num_workers=0,
            )
        assert dm is not None
        assert dm.train_dataloader() is not None

        # check training data
        data = next(iter(dm.train_dataloader()))
        imgs, labels = data[DefaultDataKeys.INPUT], data[DefaultDataKeys.TARGET]
        assert imgs.shape == (2, 3, 196, 196)
        assert labels.shape == (2, 196, 196)

    def test_map_labels(self, tmpdir):
        tmp_dir = Path(tmpdir)

        # create random dummy data

        images = [
            str(tmp_dir / "img1.png"),
            str(tmp_dir / "img2.png"),
            str(tmp_dir / "img3.png"),
        ]

        targets = [
            str(tmp_dir / "labels_img1.png"),
            str(tmp_dir / "labels_img2.png"),
            str(tmp_dir / "labels_img3.png"),
        ]

        labels_map: Dict[int, Tuple[int, int, int]] = {
            0: [0, 0, 0],
            1: [255, 255, 255],
        }

        num_classes: int = len(labels_map.keys())
        img_size: Tuple[int, int] = (196, 196)
        create_random_data(images, targets, img_size, num_classes)

        # instantiate the data module

        dm = SemanticSegmentationData.from_files(
            train_files=images,
            train_targets=targets,
            val_files=images,
            val_targets=targets,
            batch_size=2,
            num_workers=0,
        )
        assert dm is not None
        assert dm.train_dataloader() is not None

        # disable visualisation for testing
        assert dm.data_fetcher.block_viz_window is True
        dm.set_block_viz_window(False)
        assert dm.data_fetcher.block_viz_window is False

        dm.set_labels_map(labels_map)
        dm.show_train_batch("load_sample")
        dm.show_train_batch("to_tensor_transform")

        # check training data
        data = next(iter(dm.train_dataloader()))
        imgs, labels = data[DefaultDataKeys.INPUT], data[DefaultDataKeys.TARGET]
        assert imgs.shape == (2, 3, 196, 196)
        assert labels.shape == (2, 196, 196)
        assert labels.min().item() == 0
        assert labels.max().item() == 1
        assert labels.dtype == torch.int64

        # now train with `fast_dev_run`
        model = SemanticSegmentation(num_classes=2, backbone="torchvision/fcn_resnet50")
        trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
        trainer.finetune(model, dm, strategy="freeze_unfreeze")
