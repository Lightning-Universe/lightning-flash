# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
from pathlib import Path

import pytest

from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, _IMAGE_EXTRAS_TESTING, _PIL_AVAILABLE
from flash.image.detection.data import ObjectDetectionData

if _PIL_AVAILABLE:
    from PIL import Image

if _FIFTYONE_AVAILABLE:
    import fiftyone as fo


def _create_dummy_coco_json(dummy_json_path):
    dummy_json = {
        "images": [
            {
                "id": 0,
                "width": 1920,
                "height": 1080,
                "file_name": "sample_one.png",
            },
            {
                "id": 1,
                "width": 1920,
                "height": 1080,
                "file_name": "sample_two.png",
            },
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 0,
                "category_id": 0,
                "area": 150,
                "bbox": [30, 40, 20, 20],
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 0,
                "area": 240,
                "bbox": [50, 100, 280, 15],
                "iscrowd": 0,
            },
            {
                "id": 3,
                "image_id": 1,
                "category_id": 0,
                "area": 170,
                "bbox": [230, 130, 90, 180],
                "iscrowd": 0,
            },
        ],
        "categories": [
            {
                "id": 0,
                "name": "person",
                "supercategory": "person",
            }
        ],
    }

    with open(dummy_json_path, "w") as fp:
        json.dump(dummy_json, fp)


def _create_synth_coco_dataset(tmpdir):
    train_dir = Path(tmpdir / "train")
    train_dir.mkdir()

    (train_dir / "images").mkdir()
    Image.new("RGB", (1920, 1080)).save(train_dir / "images" / "sample_one.png")
    Image.new("RGB", (1920, 1080)).save(train_dir / "images" / "sample_two.png")

    (train_dir / "annotations").mkdir()
    dummy_json = train_dir / "annotations" / "sample.json"

    train_folder = os.fspath(Path(train_dir / "images"))
    coco_ann_path = os.fspath(dummy_json)
    _create_dummy_coco_json(coco_ann_path)

    return train_folder, coco_ann_path


def _create_synth_folders_dataset(tmpdir):
    predict = Path(tmpdir / "predict")
    predict.mkdir()

    (predict / "images").mkdir()
    Image.new("RGB", (224, 224)).save(predict / "images" / "sample_one.png")
    Image.new("RGB", (224, 224)).save(predict / "images" / "sample_two.png")

    predict_folder = os.fspath(Path(predict / "images"))

    return predict_folder


def _create_synth_files_dataset(tmpdir):
    predict_folder = _create_synth_folders_dataset(tmpdir)

    return [os.path.join(predict_folder, f) for f in os.listdir(predict_folder)]


def _create_synth_fiftyone_dataset(tmpdir):
    img_dir = Path(tmpdir / "fo_imgs")
    img_dir.mkdir()

    Image.new("RGB", (1920, 1080)).save(img_dir / "sample_one.png")
    Image.new("RGB", (1920, 1080)).save(img_dir / "sample_two.png")

    dataset = fo.Dataset.from_dir(
        img_dir,
        dataset_type=fo.types.ImageDirectory,
    )

    sample1 = dataset[str(img_dir / "sample_one.png")]
    sample2 = dataset[str(img_dir / "sample_two.png")]

    d1 = fo.Detection(
        label="person",
        bounding_box=[0.3, 0.4, 0.2, 0.2],
    )
    d2 = fo.Detection(
        label="person",
        bounding_box=[0.05, 0.10, 0.28, 0.15],
    )
    d3 = fo.Detection(
        label="person",
        bounding_box=[0.23, 0.14, 0.09, 0.18],
    )
    d1["iscrowd"] = 1
    d2["iscrowd"] = 0
    d3["iscrowd"] = 0

    sample1["ground_truth"] = fo.Detections(detections=[d1])
    sample2["ground_truth"] = fo.Detections(detections=[d2, d3])

    sample1.save()
    sample2.save()

    return dataset


@pytest.mark.skipif(not _IMAGE_EXTRAS_TESTING, reason="image libraries aren't installed.")
def test_image_detector_data_from_coco(tmpdir):
    train_folder, coco_ann_path = _create_synth_coco_dataset(tmpdir)

    datamodule = ObjectDetectionData.from_coco(
        train_folder=train_folder, train_ann_file=coco_ann_path, batch_size=1, transform_kwargs=dict(image_size=128)
    )

    data = next(iter(datamodule.train_dataloader()))
    sample = data[0]
    assert sample[DataKeys.INPUT].shape == (128, 128, 3)

    datamodule = ObjectDetectionData.from_coco(
        train_folder=train_folder,
        train_ann_file=coco_ann_path,
        val_folder=train_folder,
        val_ann_file=coco_ann_path,
        test_folder=train_folder,
        test_ann_file=coco_ann_path,
        batch_size=1,
        num_workers=0,
        transform_kwargs=dict(image_size=128),
    )

    data = next(iter(datamodule.val_dataloader()))
    sample = data[0]
    assert sample[DataKeys.INPUT].shape == (128, 128, 3)

    data = next(iter(datamodule.test_dataloader()))
    sample = data[0]
    assert sample[DataKeys.INPUT].shape == (128, 128, 3)


@pytest.mark.skipif(not _IMAGE_EXTRAS_TESTING, reason="image libraries aren't installed.")
@pytest.mark.skipif(not _FIFTYONE_AVAILABLE, reason="fiftyone is not installed for testing")
def test_image_detector_data_from_fiftyone(tmpdir):
    train_dataset = _create_synth_fiftyone_dataset(tmpdir)

    datamodule = ObjectDetectionData.from_fiftyone(
        train_dataset=train_dataset, batch_size=1, transform_kwargs=dict(image_size=128)
    )

    data = next(iter(datamodule.train_dataloader()))
    sample = data[0]
    assert sample[DataKeys.INPUT].shape == (128, 128, 3)

    datamodule = ObjectDetectionData.from_fiftyone(
        train_dataset=train_dataset,
        val_dataset=train_dataset,
        test_dataset=train_dataset,
        batch_size=1,
        num_workers=0,
        transform_kwargs=dict(image_size=128),
    )

    data = next(iter(datamodule.val_dataloader()))
    sample = data[0]
    assert sample[DataKeys.INPUT].shape == (128, 128, 3)

    data = next(iter(datamodule.test_dataloader()))
    sample = data[0]
    assert sample[DataKeys.INPUT].shape == (128, 128, 3)


@pytest.mark.skipif(not _IMAGE_EXTRAS_TESTING, reason="image libraries aren't installed.")
def test_image_detector_data_from_files(tmpdir):
    predict_files = _create_synth_files_dataset(tmpdir)
    datamodule = ObjectDetectionData.from_files(
        predict_files=predict_files, batch_size=1, transform_kwargs=dict(image_size=128)
    )
    data = next(iter(datamodule.predict_dataloader()))
    sample = data[0]
    assert sample[DataKeys.INPUT].shape == (128, 128, 3)


@pytest.mark.skipif(not _IMAGE_EXTRAS_TESTING, reason="image libraries aren't installed.")
def test_image_detector_data_from_folders(tmpdir):
    predict_folder = _create_synth_folders_dataset(tmpdir)
    datamodule = ObjectDetectionData.from_folders(
        predict_folder=predict_folder, batch_size=1, transform_kwargs=dict(image_size=128)
    )
    data = next(iter(datamodule.predict_dataloader()))
    sample = data[0]
    assert sample[DataKeys.INPUT].shape == (128, 128, 3)
