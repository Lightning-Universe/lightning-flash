import os
from pathlib import Path

import pytest

from flash.text.token_classification import TokenClassificationData
from flash.text.token_classification.data import LABEL_IGNORE

TEST_BACKBONE = "prajjwal1/bert-tiny"  # super small model for testing

# source: conll2003
TEST_CSV_DATA = """sentence,labels
EU rejects German call to boycott British lamb.,B-ORG O B-MISC O O O B-MISC O O
Peter Blackburn,B-PER I-PER
BRUSSELS 1996-08-22,B-LOC O
"""

TEST_JSON_DATA = """
{"sentence": "EU rejects German call to boycott British lamb.","labels": "B-ORG O B-MISC O O O B-MISC O O"}
{"sentence": "Peter Blackburn","labels": "B-PER I-PER"}
{"sentence": "BRUSSELS 1996-08-22","labels": "B-LOC O"}
"""


def csv_data(tmpdir):
    path = Path(tmpdir) / "data.csv"
    path.write_text(TEST_CSV_DATA)
    return path


def json_data(tmpdir):
    path = Path(tmpdir) / "data.json"
    path.write_text(TEST_JSON_DATA)
    return path


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
def test_from_csv(tmpdir):
    csv_path = csv_data(tmpdir)
    dm = TokenClassificationData.from_csv(
        "sentence",
        "labels",
        train_file=csv_path,
        batch_size=1,
    )
    batch = next(iter(dm.train_dataloader()))
    assert set(batch["labels"].flatten().numpy()) <= {LABEL_IGNORE, 0, 1, 2, 3, 4, 5}
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
def test_test_valid(tmpdir):
    csv_path = csv_data(tmpdir)
    dm = TokenClassificationData.from_csv(
        "sentence",
        "labels",
        backbone=TEST_BACKBONE,
        train_file=csv_path,
        val_file=csv_path,
        test_file=csv_path,
        batch_size=1,
    )
    batch = next(iter(dm.val_dataloader()))
    assert set(batch["labels"].flatten().numpy()) <= {LABEL_IGNORE, 0, 1, 2, 3, 4, 5}
    assert "input_ids" in batch

    batch = next(iter(dm.test_dataloader()))
    assert set(batch["labels"].flatten().numpy()) <= {LABEL_IGNORE, 0, 1, 2, 3, 4, 5}
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
def test_from_json(tmpdir):
    json_path = json_data(tmpdir)
    dm = TokenClassificationData.from_json(
        "sentence",
        "labels",
        backbone=TEST_BACKBONE,
        train_file=json_path,
        batch_size=1,
    )
    batch = next(iter(dm.train_dataloader()))
    assert set(batch["labels"].flatten().numpy()) <= {LABEL_IGNORE, 0, 1, 2, 3, 4, 5}
    assert "input_ids" in batch
