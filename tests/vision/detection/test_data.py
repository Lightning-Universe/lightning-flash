import json
import os
from pathlib import Path

import pytest
from PIL import Image
from pytorch_lightning.utilities import _module_available

from flash.vision.detection.data import ObjectDetectionData

_COCO_AVAILABLE = _module_available("pycocotools")


def _create_dummy_coco_json(dummy_json_path):

    dummy_json = {
        "images": [{
            "id": 0,
            'width': 1920,
            'height': 1080,
            'file_name': 'sample_one.png',
        }, {
            "id": 1,
            "width": 1920,
            "height": 1080,
            "file_name": "sample_two.png",
        }],
        "annotations": [{
            "id": 1,
            "image_id": 0,
            "category_id": 0,
            "area": 150,
            "bbox": [30, 40, 20, 20],
            "iscrowd": 0,
        }, {
            "id": 2,
            "image_id": 1,
            "category_id": 0,
            "area": 240,
            "bbox": [50, 100, 280, 15],
            "iscrowd": 0,
        }, {
            "id": 3,
            "image_id": 1,
            "category_id": 0,
            "area": 170,
            "bbox": [230, 130, 90, 180],
            "iscrowd": 0,
        }],
        "categories": [{
            "id": 0,
            "name": "person",
            "supercategory": "person",
        }]
    }

    with open(dummy_json_path, "w") as fp:
        json.dump(dummy_json, fp)


def _create_synth_coco_dataset(tmpdir):
    train_dir = Path(tmpdir / "train")
    train_dir.mkdir()

    (train_dir / "images").mkdir()
    Image.new('RGB', (1920, 1080)).save(train_dir / "images" / "sample_one.png")
    Image.new('RGB', (1920, 1080)).save(train_dir / "images" / "sample_two.png")

    (train_dir / "annotations").mkdir()
    dummy_json = train_dir / "annotations" / "sample.json"

    train_folder = os.fspath(Path(train_dir / "images"))
    coco_ann_path = os.fspath(dummy_json)
    _create_dummy_coco_json(coco_ann_path)

    return train_folder, coco_ann_path


@pytest.mark.skipif(not _COCO_AVAILABLE, reason="pycocotools is not installed for testing")
def test_image_detector_data_from_coco(tmpdir):

    train_folder, coco_ann_path = _create_synth_coco_dataset(tmpdir)

    datamodule = ObjectDetectionData.from_coco(train_folder=train_folder, train_ann_file=coco_ann_path, batch_size=1)

    data = next(iter(datamodule.train_dataloader()))
    imgs, labels = data

    assert len(imgs) == 1
    assert imgs[0].shape == (3, 1080, 1920)
    assert len(labels) == 1
    assert list(labels[0].keys()) == ['boxes', 'labels', 'image_id', 'area', 'iscrowd']

    assert datamodule.val_dataloader() is None
    assert datamodule.test_dataloader() is None

    datamodule = ObjectDetectionData.from_coco(
        train_folder=train_folder,
        train_ann_file=coco_ann_path,
        valid_folder=train_folder,
        valid_ann_file=coco_ann_path,
        test_folder=train_folder,
        test_ann_file=coco_ann_path,
        batch_size=1,
        num_workers=0
    )

    data = next(iter(datamodule.val_dataloader()))
    imgs, labels = data

    assert len(imgs) == 1
    assert imgs[0].shape == (3, 1080, 1920)
    assert len(labels) == 1
    assert list(labels[0].keys()) == ['boxes', 'labels', 'image_id', 'area', 'iscrowd']

    data = next(iter(datamodule.test_dataloader()))
    imgs, labels = data

    assert len(imgs) == 1
    assert imgs[0].shape == (3, 1080, 1920)
    assert len(labels) == 1
    assert list(labels[0].keys()) == ['boxes', 'labels', 'image_id', 'area', 'iscrowd']
