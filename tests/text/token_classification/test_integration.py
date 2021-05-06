import os
from pathlib import Path

import pytest
from pytorch_lightning import Trainer

from flash.text import TokenClassificationData, TokenClassifier

TEST_BACKBONE = "prajjwal1/bert-tiny"  # super small model for testing

# source: conll2003
TEST_CSV_DATA = """sentence,labels
EU rejects German call to boycott British lamb.,B-ORG O B-MISC O O O B-MISC O O
Peter Blackburn,B-PER I-PER
BRUSSELS 1996-08-22,B-LOC O
"""
NUM_CLASSES = 6


def csv_data(tmpdir):
    path = Path(tmpdir) / "data.csv"
    path.write_text(TEST_CSV_DATA)
    return path


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
def test_token_classification(tmpdir):
    csv_path = csv_data(tmpdir)
    data = TokenClassificationData.from_csv(
        "sentence",
        "labels",
        backbone=TEST_BACKBONE,
        train_file=csv_path,
        num_workers=0,
        batch_size=2,
    )

    model = TokenClassifier(NUM_CLASSES, TEST_BACKBONE)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, datamodule=data)
