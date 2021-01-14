import os
from imp import reload
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from PIL import Image

from pl_flash.text import TextClassificationData

TEST_BACKBONE = "prajjwal1/bert-tiny"  # super small model for testing

TEST_CSV_DATA = """sentence,label
this is a sentence one,0
this is a sentence two,1
this is a sentence three,0
"""

TEST_JSON_DATA = """
{"sentence": "this is a sentence one","lab":0}
{"sentence": "this is a sentence two","lab":1}
{"sentence": "this is a sentence three","lab":0}
"""


def csv_data(tmpdir):
    path = Path(tmpdir) / "data.csv"
    path.write_text(TEST_CSV_DATA)
    return path


def json_data(tmpdir):
    path = Path(tmpdir) / "data.json"
    path.write_text(TEST_JSON_DATA)
    return path


def test_from_csv(tmpdir):
    if os.name == "nt":
        # TODO: huggingface stuff timing out on windows
        return True
    csv_path = csv_data(tmpdir)
    dm = TextClassificationData.from_files(
        backbone=TEST_BACKBONE,
        train_file=csv_path,
        text_field="sentence",
        label_field="label",
    )
    batch = next(iter(dm.train_dataloader()))
    assert batch["labels"].item() in [0, 1]
    assert "input_ids" in batch


def test_test_valid(tmpdir):
    if os.name == "nt":
        # TODO: huggingface stuff timing out on windows
        return True
    csv_path = csv_data(tmpdir)
    dm = TextClassificationData.from_files(
        backbone=TEST_BACKBONE,
        train_file=csv_path,
        valid_file=csv_path,
        test_file=csv_path,
        text_field="sentence",
        label_field="label",
    )
    batch = next(iter(dm.val_dataloader()))
    assert batch["labels"].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.test_dataloader()))
    assert batch["labels"].item() in [0, 1]
    assert "input_ids" in batch


def test_from_json(tmpdir):
    if os.name == "nt":
        # TODO: huggingface stuff timing out on windows
        return True
    json_path = json_data(tmpdir)
    dm = TextClassificationData.from_files(
        backbone=TEST_BACKBONE,
        train_file=json_path,
        text_field="sentence",
        label_field="lab",
        filetype="json",
    )
    batch = next(iter(dm.train_dataloader()))
    assert batch["labels"].item() in [0, 1]
    assert "input_ids" in batch
