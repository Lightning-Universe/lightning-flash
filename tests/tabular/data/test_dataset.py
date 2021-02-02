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
import numpy as np
import pandas as pd

from flash.tabular.classification.data.dataset import PandasDataset

TEST_DF = pd.DataFrame(
    data={
        "category": [0, 1, 2, 1, 0, 2],
        "scalar_a": [0.0, 1.0, 2.0, 3.0, 2.0, 5.0],
        "scalar_b": [5.0, 4.0, 3.0, 2.0, 2.0, 1.0],
        "label": [0, 1, 0, 1, 0, 1],
    }
)

TEST_DF_NO_NUM = pd.DataFrame(data={
    "category": [0, 1, 2, 1, 0, 2],
    "label": [0, 1, 0, 1, 0, 1],
})

TEST_DF_NO_CAT = pd.DataFrame(data={
    "category": [0, 1, 2, 1, 0, 2],
    "label": [0, 1, 0, 1, 0, 1],
})


def test_pandas():
    df = TEST_DF.copy()
    ds = PandasDataset(
        df,
        cat_cols=["category"],
        num_cols=["scalar_a", "scalar_b"],
        target_col="label",
        regression=False,
    )
    assert len(ds) == 6
    (cat, num), target = ds[0]
    assert cat == np.array([0])
    assert np.allclose(num, np.array([0.0, 5.0]))
    assert target == 0


def test_pandas_no_cat():
    df = TEST_DF.copy()
    ds = PandasDataset(
        df,
        cat_cols=[],
        num_cols=["scalar_a", "scalar_b"],
        target_col="label",
        regression=False,
    )
    assert len(ds) == 6
    (cat, num), target = ds[0]
    assert cat.size == 0
    assert np.allclose(num, np.array([0.0, 5.0]))
    assert target == 0


def test_pandas_no_num():
    df = TEST_DF.copy()
    ds = PandasDataset(
        df,
        cat_cols=["category"],
        num_cols=[],
        target_col="label",
        regression=False,
    )
    assert len(ds) == 6
    (cat, num), target = ds[0]
    assert cat == np.array([0])
    assert num.size == 0
    assert target == 0
