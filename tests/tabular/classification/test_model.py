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
import re
from unittest import mock

import pandas as pd
import pytest
import torch
from pytorch_lightning import Trainer

from flash.__main__ import main
from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _TABULAR_AVAILABLE
from flash.tabular.classification.data import TabularClassificationData
from flash.tabular.classification.model import TabularClassifier
from tests.helpers.utils import _SERVE_TESTING, _TABULAR_TESTING

# ======== Mock functions ========


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_num=16, num_cat=16):
        super().__init__()
        self.num_num = num_num
        self.num_cat = num_cat

    def __getitem__(self, index):
        target = torch.randint(0, 10, size=(1,)).item()
        cat_vars = torch.randint(0, 10, size=(self.num_cat,))
        num_vars = torch.rand(self.num_num)
        return {DataKeys.INPUT: (cat_vars, num_vars), DataKeys.TARGET: target}

    def __len__(self) -> int:
        return 100


# ==============================


@pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular libraries aren't installed.")
@pytest.mark.parametrize("backbone", ["tabnet", "tabtransformer", "fttransformer", "autoint",
                                      "node", "category_embedding"])
def test_init_train(backbone, tmpdir):
    train_dl = torch.utils.data.DataLoader(DummyDataset(), batch_size=16)
    data_properties = {"embedding_dims": [(10, 32) for _ in range(16)],
                "categorical_cols": list(range(16)),
                "categorical_cardinality": [10 for _ in range(16)],
                "categorical_dim": 16,
                "continuous_dim": 16,
                "output_dim": 10,
                "backbone": backbone
                }

    model = TabularClassifier(**data_properties)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, train_dl)


@pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular libraries aren't installed.")
@pytest.mark.parametrize("backbone", ["tabnet", "tabtransformer", "fttransformer", "autoint",
                                      "node", "category_embedding"])
def test_init_train_no_num(backbone, tmpdir):
    train_dl = torch.utils.data.DataLoader(DummyDataset(num_num=0), batch_size=16)
    data_properties = {"embedding_dims": [(10, 32) for _ in range(16)],
                       "categorical_cols": list(range(16)),
                       "categorical_cardinality": [10 for _ in range(16)],
                       "categorical_dim": 16,
                       "continuous_dim": 0,
                       "output_dim": 10,
                       "backbone": backbone
                       }

    model = TabularClassifier(**data_properties)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, train_dl)


@pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular libraries aren't installed.")
@pytest.mark.parametrize("backbone", ["tabnet", "tabtransformer", "autoint",
                                      "node", "category_embedding"])
def test_init_train_no_cat(backbone, tmpdir):
    train_dl = torch.utils.data.DataLoader(DummyDataset(num_cat=0), batch_size=16)
    data_properties = {"embedding_dims": [],
                       "categorical_cols": [],
                       "categorical_cardinality": [],
                       "categorical_dim": 0,
                       "continuous_dim": 16,
                       "output_dim": 10,
                       "backbone": backbone
                       }

    model = TabularClassifier(**data_properties)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, train_dl)


@pytest.mark.skipif(_TABULAR_AVAILABLE, reason="tabular libraries are installed.")
def test_module_import_error(tmpdir):
    data_properties = {"embedding_dims": [],
                       "categorical_cols": [],
                       "categorical_cardinality": [],
                       "categorical_dim": 0,
                       "continuous_dim": 16,
                       "output_dim": 10,
                       "backbone": "tabnet"
                       }
    with pytest.raises(ModuleNotFoundError, match="[tabular]"):
        TabularClassifier(**data_properties)


@pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular libraries aren't installed.")
@pytest.mark.parametrize("backbone", ["tabnet", "tabtransformer", "fttransformer", "autoint",
                                      "node", "category_embedding"])
def test_jit(backbone, tmpdir):
    data_properties = {"embedding_dims": [(10, 32) for _ in range(4)],
                       "categorical_cols": list(range(4)),
                       "categorical_cardinality": [10 for _ in range(4)],
                       "categorical_dim": 4,
                       "continuous_dim": 4,
                       "output_dim": 10,
                       "backbone": backbone
                       }
    model = TabularClassifier(**data_properties)
    model.eval()

    # torch.jit.script doesn't work with tabnet
    batch = {
        "continuous": torch.rand(1, 4),
        "categorical": torch.randint(0, 10, size=(1, 4)),
    }
    model = torch.jit.trace(model, batch, check_trace=False)

    # TODO: torch.jit.save doesn't work with tabnet
    # path = os.path.join(tmpdir, "test.pt")
    # torch.jit.save(model, path)
    # model = torch.jit.load(path)

    out = model(batch)
    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([1, 10])


@pytest.mark.skipif(not _SERVE_TESTING, reason="serve libraries aren't installed.")
@pytest.mark.parametrize("backbone", ["tabnet", "tabtransformer", "fttransformer", "autoint",
                                      "node", "category_embedding"])
@mock.patch("flash._IS_TESTING", True)
def test_serve(backbone):
    train_data = {"num_col": [1.4, 2.5], "cat_col": ["positive", "negative"], "target": [1, 2]}
    datamodule = TabularClassificationData.from_data_frame(
        "cat_col",
        "num_col",
        "target",
        train_data_frame=pd.DataFrame.from_dict(train_data),
        batch_size=1,
    )
    model = TabularClassifier.from_data(datamodule=datamodule, backbone=backbone)
    # TODO: Currently only servable once a input_transform has been attached
    model._input_transform = datamodule.input_transform
    model.eval()
    model.serve(parameters=datamodule.parameters)


@pytest.mark.skipif(_TABULAR_AVAILABLE, reason="tabular libraries are installed.")
def test_load_from_checkpoint_dependency_error():
    with pytest.raises(ModuleNotFoundError, match=re.escape("'lightning-flash[tabular]'")):
        TabularClassifier.load_from_checkpoint("not_a_real_checkpoint.pt")


# @pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular libraries aren't installed.")
# def test_cli():
#     cli_args = ["flash", "tabular_classification", "--trainer.fast_dev_run", "True"]
#     with mock.patch("sys.argv", cli_args):
#         try:
#             main()
#         except SystemExit:
#             pass
