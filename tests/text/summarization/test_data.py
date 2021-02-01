import os
from pathlib import Path

from flash.text import SummarizationData

TEST_BACKBONE = "sshleifer/tiny-mbart"  # super small model for testing

TEST_CSV_DATA = """input,target
this is a sentence one,this is a translated sentence one
this is a sentence two,this is a translated sentence two
this is a sentence three,this is a translated sentence three
"""

TEST_JSON_DATA = """
{"input": "this is a sentence one","target":"this is a translated sentence one"}
{"input": "this is a sentence two","target":"this is a translated sentence two"}
{"input": "this is a sentence three","target":"this is a translated sentence three"}
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
    dm = SummarizationData.from_files(
        backbone=TEST_BACKBONE,
        train_file=csv_path,
        input="input",
        target="target",
        batch_size=1
    )
    batch = next(iter(dm.train_dataloader()))
    assert "labels" in batch
    assert "input_ids" in batch


def test_from_files(tmpdir):
    if os.name == "nt":
        # TODO: huggingface stuff timing out on windows
        return True
    csv_path = csv_data(tmpdir)
    dm = SummarizationData.from_files(
        backbone=TEST_BACKBONE,
        train_file=csv_path,
        valid_file=csv_path,
        test_file=csv_path,
        input="input",
        target="target",
        batch_size=1
    )
    batch = next(iter(dm.val_dataloader()))
    assert "labels" in batch
    assert "input_ids" in batch

    batch = next(iter(dm.test_dataloader()))
    assert "labels" in batch
    assert "input_ids" in batch


def test_from_json(tmpdir):
    if os.name == "nt":
        # TODO: huggingface stuff timing out on windows
        return True
    json_path = json_data(tmpdir)
    dm = SummarizationData.from_files(
        backbone=TEST_BACKBONE,
        train_file=json_path,
        input="input",
        target="target",
        filetype="json",
        batch_size=1
    )
    batch = next(iter(dm.train_dataloader()))
    assert "labels" in batch
    assert "input_ids" in batch
