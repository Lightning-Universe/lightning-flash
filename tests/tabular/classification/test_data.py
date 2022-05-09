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
import pytest

from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _TABULAR_TESTING

if _TABULAR_TESTING:
    import pandas as pd

    from flash.tabular import TabularClassificationData
    from flash.tabular.classification.utils import _categorize, _compute_normalization, _generate_codes, _normalize

    TEST_DICT_1 = {
        "category": ["a", "b", "c", "a", None, "c"],
        "scalar_a": [0.0, 1.0, 2.0, 3.0, None, 5.0],
        "scalar_b": [5.0, 4.0, 3.0, 2.0, None, 1.0],
        "label": [0, 1, 0, 1, 0, 1],
    }

    TEST_DICT_2 = {
        "category": ["d", "e", "f"],
        "scalar_a": [0.0, 1.0, 2.0],
        "scalar_b": [0.0, 4.0, 2.0],
        "label": [0, 1, 0],
    }

    TEST_DF_1 = pd.DataFrame(data=TEST_DICT_1)
    TEST_DF_2 = pd.DataFrame(data=TEST_DICT_2)


@pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular dependencies are required")
def test_categorize():
    codes = _generate_codes(TEST_DF_1, ["category"])
    assert codes == {"category": ["a", "b", "c"]}

    df = _categorize(TEST_DF_1, ["category"], codes)
    assert list(df["category"]) == [1, 2, 3, 1, 0, 3]

    df = _categorize(TEST_DF_2, ["category"], codes)
    assert list(df["category"]) == [0, 0, 0]


@pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular dependencies are required")
def test_normalize():
    num_input = ["scalar_a", "scalar_b"]
    mean, std = _compute_normalization(TEST_DF_1, num_input)
    df = _normalize(TEST_DF_1, num_input, mean, std)
    assert np.allclose(df[num_input].mean(), 0.0)


@pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular dependencies are required")
def test_embedding_sizes():
    self = Mock()

    self.cat_dims = [4]
    # use __get__ to test property with mocked self
    es = TabularClassificationData.embedding_sizes.__get__(self)  # pylint: disable=E1101
    assert es == [(4, 16)]

    self.cat_dims = []
    # use __get__ to test property with mocked self
    es = TabularClassificationData.embedding_sizes.__get__(self)  # pylint: disable=E1101
    assert es == []

    self.cat_dims = [100_000, 1_000_000]
    # use __get__ to test property with mocked self
    es = TabularClassificationData.embedding_sizes.__get__(self)  # pylint: disable=E1101
    assert es == [(100_000, 17), (1_000_000, 31)]


@pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular dependencies are required")
def test_categorical_target(tmpdir):
    train_data_frame = TEST_DF_1.copy()
    val_data_frame = TEST_DF_2.copy()
    test_data_frame = TEST_DF_2.copy()
    for df in [train_data_frame, val_data_frame, test_data_frame]:
        # change int label to string
        df["label"] = df["label"].astype(str)

    dm = TabularClassificationData.from_data_frame(
        categorical_fields=["category"],
        numerical_fields=["scalar_a", "scalar_b"],
        target_fields="label",
        train_data_frame=train_data_frame,
        val_data_frame=val_data_frame,
        test_data_frame=test_data_frame,
        num_workers=0,
        batch_size=1,
    )
    for dl in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
        data = next(iter(dl))
        (cat, num) = data[DataKeys.INPUT]
        target = data[DataKeys.TARGET]
        assert cat.shape == (1, 1)
        assert num.shape == (1, 2)
        assert target.shape == (1,)


@pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular dependencies are required")
def test_from_data_frame(tmpdir):
    train_data_frame = TEST_DF_1.copy()
    val_data_frame = TEST_DF_2.copy()
    test_data_frame = TEST_DF_2.copy()
    dm = TabularClassificationData.from_data_frame(
        categorical_fields=["category"],
        numerical_fields=["scalar_a", "scalar_b"],
        target_fields="label",
        train_data_frame=train_data_frame,
        val_data_frame=val_data_frame,
        test_data_frame=test_data_frame,
        num_workers=0,
        batch_size=1,
    )
    for dl in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
        data = next(iter(dl))
        (cat, num) = data[DataKeys.INPUT]
        target = data[DataKeys.TARGET]
        assert cat.shape == (1, 1)
        assert num.shape == (1, 2)
        assert target.shape == (1,)


@pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular dependencies are required")
def test_from_csv(tmpdir):
    train_csv = Path(tmpdir) / "train.csv"
    val_csv = test_csv = Path(tmpdir) / "valid.csv"
    TEST_DF_1.to_csv(train_csv)
    TEST_DF_2.to_csv(val_csv)
    TEST_DF_2.to_csv(test_csv)

    dm = TabularClassificationData.from_csv(
        categorical_fields=["category"],
        numerical_fields=["scalar_a", "scalar_b"],
        target_fields="label",
        train_file=str(train_csv),
        val_file=str(val_csv),
        test_file=str(test_csv),
        num_workers=0,
        batch_size=1,
    )
    for dl in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
        data = next(iter(dl))
        (cat, num) = data[DataKeys.INPUT]
        target = data[DataKeys.TARGET]
        assert cat.shape == (1, 1)
        assert num.shape == (1, 2)
        assert target.shape == (1,)


@pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular dependencies are required")
def test_from_dicts():
    dm = TabularClassificationData.from_dicts(
        categorical_fields=["category"],
        numerical_fields=["scalar_a", "scalar_b"],
        target_fields="label",
        train_dict=TEST_DICT_1,
        val_dict=TEST_DICT_2,
        test_dict=TEST_DICT_2,
        num_workers=0,
        batch_size=1,
    )
    for dl in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
        data = next(iter(dl))
        (cat, num) = data[DataKeys.INPUT]
        target = data[DataKeys.TARGET]
        assert cat.shape == (1, 1)
        assert num.shape == (1, 2)
        assert target.shape == (1,)


@pytest.mark.skipif(not _TABULAR_TESTING, reason="tabular dependencies are required")
def test_empty_inputs():
    train_data_frame = TEST_DF_1.copy()
    with pytest.raises(RuntimeError):
        TabularClassificationData.from_data_frame(
            numerical_fields=None,
            categorical_fields=None,
            target_fields="label",
            train_data_frame=train_data_frame,
            num_workers=0,
            batch_size=1,
        )
