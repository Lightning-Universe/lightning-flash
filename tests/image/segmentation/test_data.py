import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash import Trainer
from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, _IMAGE_AVAILABLE, _MATPLOTLIB_AVAILABLE, _PIL_AVAILABLE
from flash.image import SemanticSegmentation, SemanticSegmentationData, SemanticSegmentationInputTransform
from tests.helpers.utils import _IMAGE_TESTING

if _PIL_AVAILABLE:
    from PIL import Image

if _FIFTYONE_AVAILABLE:
    import fiftyone as fo


def build_checkboard(n, m, k=8):
    x = np.zeros((n, m))
    x[k :: k * 2, ::k] = 1
    x[:: k * 2, k :: k * 2] = 1
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


class TestSemanticSegmentationInputTransform:
    @staticmethod
    @pytest.mark.xfail(reaspn="parameters are marked as optional but it returns Misconficg error.")
    def test_smoke():
        prep = SemanticSegmentationInputTransform(num_classes=1)
        assert prep is not None


class TestSemanticSegmentationData:
    @staticmethod
    @pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
    def test_smoke():
        dm = SemanticSegmentationData()
        assert dm is not None

    @staticmethod
    @pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
    def test_from_folders(tmpdir):
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
        img_size: Tuple[int, int] = (128, 128)
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
            num_classes=num_classes,
        )
        assert dm is not None
        assert dm.train_dataloader() is not None
        assert dm.val_dataloader() is not None
        assert dm.test_dataloader() is not None

        # check training data
        data = next(iter(dm.train_dataloader()))
        imgs, labels = data[DataKeys.INPUT], data[DataKeys.TARGET]
        assert imgs.shape == (2, 3, 128, 128)
        assert labels.shape == (2, 128, 128)

        # check val data
        data = next(iter(dm.val_dataloader()))
        imgs, labels = data[DataKeys.INPUT], data[DataKeys.TARGET]
        assert imgs.shape == (2, 3, 128, 128)
        assert labels.shape == (2, 128, 128)

        # check test data
        data = next(iter(dm.test_dataloader()))
        imgs, labels = data[DataKeys.INPUT], data[DataKeys.TARGET]
        assert imgs.shape == (2, 3, 128, 128)
        assert labels.shape == (2, 128, 128)

    @staticmethod
    @pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
    def test_from_folders_warning(tmpdir):
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
        img_size: Tuple[int, int] = (128, 128)
        create_random_data(images, targets, img_size, num_classes)

        # instantiate the data module

        with pytest.warns(UserWarning, match="Found inconsistent files"):
            dm = SemanticSegmentationData.from_folders(
                train_folder=str(tmp_dir / "images"),
                train_target_folder=str(tmp_dir / "targets"),
                batch_size=1,
                num_workers=0,
                num_classes=num_classes,
            )
        assert dm is not None
        assert dm.train_dataloader() is not None

        # check training data
        data = next(iter(dm.train_dataloader()))
        imgs, labels = data[DataKeys.INPUT], data[DataKeys.TARGET]
        assert imgs.shape == (1, 3, 128, 128)
        assert labels.shape == (1, 128, 128)

    @staticmethod
    @pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
    def test_from_files(tmpdir):
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
        img_size: Tuple[int, int] = (128, 128)
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
            num_classes=num_classes,
        )
        assert dm is not None
        assert dm.train_dataloader() is not None
        assert dm.val_dataloader() is not None
        assert dm.test_dataloader() is not None

        # check training data
        data = next(iter(dm.train_dataloader()))
        imgs, labels = data[DataKeys.INPUT], data[DataKeys.TARGET]
        assert imgs.shape == (2, 3, 128, 128)
        assert labels.shape == (2, 128, 128)

        # check val data
        data = next(iter(dm.val_dataloader()))
        imgs, labels = data[DataKeys.INPUT], data[DataKeys.TARGET]
        assert imgs.shape == (2, 3, 128, 128)
        assert labels.shape == (2, 128, 128)

        # check test data
        data = next(iter(dm.test_dataloader()))
        imgs, labels = data[DataKeys.INPUT], data[DataKeys.TARGET]
        assert imgs.shape == (2, 3, 128, 128)
        assert labels.shape == (2, 128, 128)

    @staticmethod
    @pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
    def test_from_files_warning(tmpdir):
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
        img_size: Tuple[int, int] = (128, 128)
        create_random_data(images, targets, img_size, num_classes)

        # instantiate the data module

        with pytest.raises(MisconfigurationException, match="The number of input files"):
            SemanticSegmentationData.from_files(
                train_files=images,
                train_targets=targets + [str(tmp_dir / "labels_img4.png")],
                batch_size=2,
                num_workers=0,
                num_classes=num_classes,
            )

    @staticmethod
    @pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed.")
    @pytest.mark.skipif(not _FIFTYONE_AVAILABLE, reason="fiftyone is not installed for testing")
    def test_from_fiftyone(tmpdir):
        tmp_dir = Path(tmpdir)

        # create random dummy data

        images = [
            str(tmp_dir / "img1.png"),
            str(tmp_dir / "img2.png"),
            str(tmp_dir / "img3.png"),
        ]

        num_classes: int = 2
        img_size: Tuple[int, int] = (128, 128)

        for img_file in images:
            _rand_image(img_size).save(img_file)

        targets = [np.array(_rand_labels(img_size, num_classes)) for _ in range(3)]

        dataset = fo.Dataset.from_dir(
            str(tmp_dir),
            dataset_type=fo.types.ImageDirectory,
        )

        for idx, sample in enumerate(dataset):
            sample["ground_truth"] = fo.Segmentation(mask=targets[idx][:, :, 0])
            sample.save()

        # instantiate the data module

        dm = SemanticSegmentationData.from_fiftyone(
            train_dataset=dataset,
            val_dataset=dataset,
            test_dataset=dataset,
            predict_dataset=dataset,
            batch_size=2,
            num_workers=0,
            num_classes=num_classes,
        )
        assert dm is not None
        assert dm.train_dataloader() is not None
        assert dm.val_dataloader() is not None
        assert dm.test_dataloader() is not None

        # check training data
        data = next(iter(dm.train_dataloader()))
        imgs, labels = data[DataKeys.INPUT], data[DataKeys.TARGET]
        assert imgs.shape == (2, 3, 128, 128)
        assert labels.shape == (2, 128, 128)

        # check val data
        data = next(iter(dm.val_dataloader()))
        imgs, labels = data[DataKeys.INPUT], data[DataKeys.TARGET]
        assert imgs.shape == (2, 3, 128, 128)
        assert labels.shape == (2, 128, 128)

        # check test data
        data = next(iter(dm.test_dataloader()))
        imgs, labels = data[DataKeys.INPUT], data[DataKeys.TARGET]
        assert imgs.shape == (2, 3, 128, 128)
        assert labels.shape == (2, 128, 128)

        # check predict data
        data = next(iter(dm.predict_dataloader()))
        imgs = data[DataKeys.INPUT]
        assert imgs.shape == (2, 3, 128, 128)

    @staticmethod
    @pytest.mark.skipif(not _IMAGE_AVAILABLE, reason="image libraries aren't installed.")
    @pytest.mark.skipif(not _MATPLOTLIB_AVAILABLE, reason="matplotlib isn't installed.")
    def test_map_labels(tmpdir):
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
        img_size: Tuple[int, int] = (128, 128)
        create_random_data(images, targets, img_size, num_classes)

        # instantiate the data module

        dm = SemanticSegmentationData.from_files(
            train_files=images,
            train_targets=targets,
            val_files=images,
            val_targets=targets,
            batch_size=2,
            num_workers=0,
            num_classes=num_classes,
        )
        assert dm is not None
        assert dm.train_dataloader() is not None

        # disable visualisation for testing
        assert dm.data_fetcher.block_viz_window is True
        dm.set_block_viz_window(False)
        assert dm.data_fetcher.block_viz_window is False

        dm.show_train_batch("load_sample")
        dm.show_train_batch("per_sample_transform")

        # check training data
        data = next(iter(dm.train_dataloader()))
        imgs, labels = data[DataKeys.INPUT], data[DataKeys.TARGET]
        assert imgs.shape == (2, 3, 128, 128)
        assert labels.shape == (2, 128, 128)
        assert labels.min().item() == 0
        assert labels.max().item() == 1
        assert labels.dtype == torch.int64

        # now train with `fast_dev_run`
        model = SemanticSegmentation(num_classes=2, backbone="resnet50", head="fpn")
        trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
        trainer.finetune(model, dm, strategy="freeze")
