import os
from pathlib import Path

from pytorch_lightning import Trainer

from flash.text import TextClassificationData, TextClassifier

TEST_BACKBONE = "prajjwal1/bert-tiny"  # super small model for testing

TEST_CSV_DATA = """sentence,label
this is a sentence one,0
this is a sentence two,1
this is a sentence three,0
"""


def csv_data(tmpdir):
    path = Path(tmpdir) / "data.csv"
    path.write_text(TEST_CSV_DATA)
    return path


def test_classification(tmpdir):
    if os.name == "nt":
        # TODO: huggingface stuff timing out on windows
        return True

    csv_path = csv_data(tmpdir)

    data = TextClassificationData.from_files(
        backbone=TEST_BACKBONE,
        train_file=csv_path,
        input="sentence",
        target="label",
        num_workers=0,
        batch_size=2,
    )
    model = TextClassifier(2, TEST_BACKBONE)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, datamodule=data)
