import os
import platform
from argparse import ArgumentParser
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch

from pl_flash.text import TextClassificationData, TextClassifier
from pl_flash_examples.text_classification import train_text_classification

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
        text_field="sentence",
        label_field="label",
        num_workers=0,
    )
    model = TextClassifier(2, backbone=TEST_BACKBONE)
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, data)


@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed training is not supported on Windows")
def test_train_text_classification(tmpdir):
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    cmd = "--max_epochs 1 --fast_dev_run 1 --num_processes 2 --accelerator ddp_cpu"
    args = parser.parse_args(cmd.split(" "))
    train_text_classification(args)
