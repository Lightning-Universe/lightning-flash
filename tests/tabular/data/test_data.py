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
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from flash.tabular import TabularData
from flash.tabular.classification.data.dataset import _categorize, _normalize

TEST_DF_1 = pd.DataFrame(
    data={
        "category": ["a", "b", "c", "a", None, "c"],
        "scalar_a": [0.0, 1.0, 2.0, 3.0, None, 5.0],
        "scalar_b": [5.0, 4.0, 3.0, 2.0, None, 1.0],
        "label": [0, 1, 0, 1, 0, 1],
    }
)

TEST_DF_2 = pd.DataFrame(
    data={
        "category": ["d", "e", "f"],
        "scalar_a": [0.0, 1.0, 2.0],
        "scalar_b": [0.0, 4.0, 2.0],
        "label": [0, 1, 0],
    }
)


def test_categorize():
    dfs, codes = _categorize([TEST_DF_1], ["category"])
    assert list(dfs[0]["category"]) == [1, 2, 3, 1, 0, 3]
    assert codes == {"category": [None, "a", "b", "c"]}

    dfs, codes = _categorize([TEST_DF_1, TEST_DF_2], ["category"])
    assert list(dfs[1]["category"]) == [4, 5, 6]
    assert codes == {"category": [None, "a", "b", "c", "d", "e", "f"]}


def test_normalize():
    num_input = ["scalar_a", "scalar_b"]
    dfs, mean_one, std_one = _normalize([TEST_DF_1], num_input)
    assert np.allclose(dfs[0][num_input].mean(), 0.0)

    _, mean_two, std_two = _normalize([TEST_DF_1, TEST_DF_2], num_input)
    assert np.allclose(mean_one, mean_two)
    assert np.allclose(std_one, std_two)


def test_emb_sizes():
    self = Mock()
    self.codes = {"category": [None, "a", "b", "c"]}
    self.cat_cols = ["category"]
    # use __get__ to test property with mocked self
    es = TabularData.emb_sizes.__get__(self)  # pylint: disable=E1101
    assert es == [(4, 16)]

    self.codes = {}
    self.cat_cols = []
    # use __get__ to test property with mocked self
    es = TabularData.emb_sizes.__get__(self)  # pylint: disable=E1101
    assert es == []

    self.codes = {"large": ["a"] * 100_000, "larger": ["b"] * 1_000_000}
    self.cat_cols = ["large", "larger"]
    # use __get__ to test property with mocked self
    es = TabularData.emb_sizes.__get__(self)  # pylint: disable=E1101
    assert es == [(100_000, 17), (1_000_000, 31)]


def test_tabular_data(tmpdir):
    train_df = TEST_DF_1.copy()
    valid_df = TEST_DF_2.copy()
    test_df = TEST_DF_2.copy()
    dm = TabularData(
        train_df,
        categorical_input=["category"],
        numerical_input=["scalar_b", "scalar_b"],
        target="label",
        valid_df=valid_df,
        test_df=test_df,
        num_workers=0,
        batch_size=1,
    )
    for dl in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
        (cat, num), target = next(iter(dl))
        assert cat.shape == (1, 1)
        assert num.shape == (1, 2)
        assert target.shape == (1, )


def test_categorical_target(tmpdir):
    train_df = TEST_DF_1.copy()
    valid_df = TEST_DF_2.copy()
    test_df = TEST_DF_2.copy()
    for df in [train_df, valid_df, test_df]:
        # change int label to string
        df["label"] = df["label"].astype(str)

    dm = TabularData(
        train_df,
        categorical_input=["category"],
        numerical_input=["scalar_b", "scalar_b"],
        target="label",
        valid_df=valid_df,
        test_df=test_df,
        num_workers=0,
        batch_size=1,
    )
    for dl in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
        (cat, num), target = next(iter(dl))
        assert cat.shape == (1, 1)
        assert num.shape == (1, 2)
        assert target.shape == (1, )


def test_from_df(tmpdir):
    train_df = TEST_DF_1.copy()
    valid_df = TEST_DF_2.copy()
    test_df = TEST_DF_2.copy()
    dm = TabularData.from_df(
        train_df,
        categorical_input=["category"],
        numerical_input=["scalar_b", "scalar_b"],
        target="label",
        valid_df=valid_df,
        test_df=test_df,
        num_workers=0,
        batch_size=1
    )
    for dl in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
        (cat, num), target = next(iter(dl))
        assert cat.shape == (1, 1)
        assert num.shape == (1, 2)
        assert target.shape == (1, )


def test_from_csv(tmpdir):
    train_csv = Path(tmpdir) / "train.csv"
    valid_csv = test_csv = Path(tmpdir) / "valid.csv"
    TEST_DF_1.to_csv(train_csv)
    TEST_DF_2.to_csv(valid_csv)
    TEST_DF_2.to_csv(test_csv)

    dm = TabularData.from_csv(
        train_csv,
        categorical_input=["category"],
        numerical_input=["scalar_b", "scalar_b"],
        target="label",
        valid_csv=valid_csv,
        test_csv=test_csv,
        num_workers=0,
        batch_size=1
    )
    for dl in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
        (cat, num), target = next(iter(dl))
        assert cat.shape == (1, 1)
        assert num.shape == (1, 2)
        assert target.shape == (1, )


def test_empty_inputs():
    train_df = TEST_DF_1.copy()
    with pytest.raises(RuntimeError):
        TabularData.from_df(
            train_df, categorical_input=None, numerical_input=None, target="label", num_workers=0, batch_size=1
        )
