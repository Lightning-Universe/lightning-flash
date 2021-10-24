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
import os
import re
from unittest import mock

import pytest
import torch

from flash import Trainer
from flash.__main__ import main
from flash.core.data.data_source import DefaultDataKeys
from flash.core.utilities.imports import _TEXT_AVAILABLE
from flash.text import TextClassifier
from flash.text.classification.data import TextClassificationPostprocess, TextClassificationPreprocess
from tests.helpers.utils import _SERVE_TESTING, _TEXT_TESTING

# ======== Mock functions ========


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return {
            "input_ids": torch.randint(1000, size=(100,)),
            "attention_mask": torch.ones(size=(100,)),
            DefaultDataKeys.TARGET: torch.randint(2, size=(1,)).item(),
        }

    def __len__(self) -> int:
        return 100


# ==============================

TEST_HF_BACKBONE = "prajjwal1/bert-tiny"  # super small model for testing


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("strategy", ("avg", "cls_token", "pooler_output"))
@pytest.mark.parametrize("pretrained", (True, False))
@pytest.mark.parametrize("vocab_size", (None, 1000))
def test_hf_backbones_init_train(tmpdir, strategy, pretrained, vocab_size):
    model = TextClassifier(
        num_classes=2,
        backbone=TEST_HF_BACKBONE,
        pretrained=pretrained,
        backbone_kwargs={"strategy": strategy, "vocab_size": vocab_size},
    )
    dl = torch.utils.data.DataLoader(DummyDataset())
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, train_dataloader=dl, val_dataloaders=dl)


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("strategy", ("avg", "cls_token", "pooler_output"))
@pytest.mark.parametrize("pretrained", (True, False))
@pytest.mark.parametrize("vocab_size", (None, 1000))
@pytest.mark.parametrize("finetune_strategy", ("no_freeze", "freeze"))
def test_hf_backbones_finetune(tmpdir, strategy, pretrained, vocab_size, finetune_strategy):
    model = TextClassifier(
        num_classes=2,
        backbone=TEST_HF_BACKBONE,
        pretrained=pretrained,
        backbone_kwargs={"strategy": strategy, "vocab_size": vocab_size},
    )
    dl = torch.utils.data.DataLoader(DummyDataset())
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.finetune(model, train_dataloader=dl, val_dataloaders=dl, strategy=finetune_strategy)


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("strategy", ("avg", "cls_token", "pooler_output"))
@pytest.mark.parametrize("pretrained", (True, False))
@pytest.mark.parametrize("vocab_size", (None, 1000))
def test_hf_backbones_test(tmpdir, strategy, pretrained, vocab_size):
    model = TextClassifier(
        num_classes=2,
        backbone=TEST_HF_BACKBONE,
        pretrained=pretrained,
        backbone_kwargs={"strategy": strategy, "vocab_size": vocab_size},
    )
    dl = torch.utils.data.DataLoader(DummyDataset())
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.test(model, dataloaders=dl)


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("strategy", ("avg", "cls_token", "pooler_output"))
@pytest.mark.parametrize("pretrained", (True, False))
@pytest.mark.parametrize("vocab_size", (None, 1000))
def test_hf_backbones_predict(tmpdir, strategy, pretrained, vocab_size):
    model = TextClassifier(
        num_classes=2,
        backbone=TEST_HF_BACKBONE,
        pretrained=pretrained,
        backbone_kwargs={"strategy": strategy, "vocab_size": vocab_size},
    )
    dl = torch.utils.data.DataLoader(DummyDataset())
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.predict(model, dataloaders=dl)


@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("strategy", ("avg", "cls_token", "pooler_output"))
@pytest.mark.parametrize("pretrained", (True, False))
@pytest.mark.parametrize("vocab_size", (None, 1000))
def test_hf_backbones_jit(tmpdir, strategy, pretrained, vocab_size):
    path = os.path.join(tmpdir, "test.pt")
    model = TextClassifier(
        num_classes=2,
        backbone=TEST_HF_BACKBONE,
        pretrained=pretrained,
        backbone_kwargs={"strategy": strategy, "vocab_size": vocab_size},
    )
    dl = torch.utils.data.DataLoader(DummyDataset())

    model.eval()

    # Huggingface bert model only supports `torch.jit.trace` with `strict=False`
    sample_input = next(iter(dl))
    model = torch.jit.trace(model, sample_input, strict=False)

    torch.jit.save(model, path)
    model = torch.jit.load(path)

    out = model(sample_input)
    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([1, 2])


@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_hf_backbones_pretrained():
    m1 = TextClassifier(num_classes=2, backbone=TEST_HF_BACKBONE, pretrained=True)
    m2 = TextClassifier(num_classes=2, backbone=TEST_HF_BACKBONE, pretrained=False)

    for (_, param1), (_, param2) in zip(m1.backbone.named_parameters(), m2.backbone.named_parameters()):
        assert torch.all(param1 != param2)


@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", (True, False))
def test_hf_backbones_reinit_word_embeddings(pretrained):
    m1 = TextClassifier(num_classes=2, backbone=TEST_HF_BACKBONE, pretrained=pretrained)
    m2 = TextClassifier(
        num_classes=2, backbone=TEST_HF_BACKBONE, pretrained=pretrained, backbone_kwargs={"vocab_size": 10}
    )

    for (name1, param1), (name2, param2) in zip(m1.backbone.named_parameters(), m2.backbone.named_parameters()):
        print(name1, name2)
        if "word_embeddings" in name1:
            assert param1.shape != param2.shape
            continue
        if pretrained:
            # when pretrained, all other params should be equal
            assert torch.all(param1 == param2)


@pytest.mark.skipif(not _SERVE_TESTING, reason="serve libraries aren't installed.")
@mock.patch("flash._IS_TESTING", True)
def test_serve():
    model = TextClassifier(2, TEST_HF_BACKBONE)
    # TODO: Currently only servable once a preprocess and postprocess have been attached
    model._preprocess = TextClassificationPreprocess(backbone=TEST_HF_BACKBONE)
    model._postprocess = TextClassificationPostprocess()
    model.eval()
    model.serve()


@pytest.mark.skipif(_TEXT_AVAILABLE, reason="text libraries are installed.")
def test_load_from_checkpoint_dependency_error():
    with pytest.raises(ModuleNotFoundError, match=re.escape("'lightning-flash[text]'")):
        TextClassifier.load_from_checkpoint("not_a_real_checkpoint.pt")


@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize(
    "cli_args",
    (
        ["flash", "text_classification", "--trainer.fast_dev_run", "True"],
        # TODO: update this to work with Pietro's new text data loading (separate PR)
        # ["flash", "text_classification", "--trainer.fast_dev_run", "True", "from_toxic"],
    ),
)
def test_cli(cli_args):
    with mock.patch("sys.argv", cli_args):
        try:
            main()
        except SystemExit:
            pass
