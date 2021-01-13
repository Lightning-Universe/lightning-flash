from pl_flash.tabular import TabularData

from pl_flash.tabular.data.data import _categorize, _normalize


import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, Mock
from imp import reload

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
    num_cols = ["scalar_a", "scalar_b"]
    dfs, mean_one, std_one = _normalize([TEST_DF_1], num_cols)
    assert np.allclose(dfs[0][num_cols].mean(), 0.0)

    _, mean_two, std_two = _normalize([TEST_DF_1, TEST_DF_2], num_cols)
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
        categorical_cols=["category"],
        numerical_cols=["scalar_b", "scalar_b"],
        target_col="label",
        valid_df=valid_df,
        test_df=test_df,
        num_workers=0,
        batch_size=1,
    )
    for dl in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
        data = next(iter(dl))
        (cat, num), target = data['x'], data['target']
        assert cat.shape == (1, 1)
        assert num.shape == (1, 2)
        assert target.shape == (1,)


def test_from_df(tmpdir):
    train_df = TEST_DF_1.copy()
    valid_df = TEST_DF_2.copy()
    test_df = TEST_DF_2.copy()
    dm = TabularData.from_df(
        train_df,
        categorical_cols=["category"],
        numerical_cols=["scalar_b", "scalar_b"],
        target_col="label",
        valid_df=valid_df,
        test_df=test_df,
        num_workers=0,
    )
    for dl in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
        data = next(iter(dl))
        (cat, num), target = data['x'], data['target']
        assert cat.shape == (1, 1)
        assert num.shape == (1, 2)
        assert target.shape == (1,)
