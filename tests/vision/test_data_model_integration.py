import platform
from argparse import ArgumentParser

import pytest
import pytorch_lightning as pl

from pl_flash_examples.generic_task import train_generic_task_on_mnist
from pl_flash_examples.torchvision_classifier import train_image_classifier


@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed training is not supported on Windows")
def test_torchvision_classifier_example(tmpdir):
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    cmd = "--max_epochs 1 --fast_dev_run 1 --num_processes 2 --accelerator ddp_cpu"
    args = parser.parse_args(cmd.split(" "))
    train_image_classifier(args)


@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed training is not supported on Windows")
def test_generic_task_example(tmpdir):
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    cmd = "--max_epochs 1 --fast_dev_run 1 --num_processes 2 --accelerator ddp_cpu"
    args = parser.parse_args(cmd.split(" "))
    train_generic_task_on_mnist(args)
