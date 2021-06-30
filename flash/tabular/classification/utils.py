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
from typing import Dict, List, Tuple

import numpy as np

from flash.core.utilities.imports import _PANDAS_AVAILABLE

if _PANDAS_AVAILABLE:
    import pandas as pd
    from pandas.core.frame import DataFrame
else:
    DataFrame = None


def _impute(dfs: List, num_cols: List) -> list:
    dfs = [df.copy() for df in dfs]
    for col in num_cols:
        for df in dfs:
            df[col] = df[col].fillna(dfs[0][col].median())
    return dfs


def _compute_normalization(df: DataFrame, num_cols: List) -> Tuple:
    return df[num_cols].mean(), df[num_cols].std()


def _normalize(dfs: List[DataFrame], num_cols: List, mean: DataFrame = None, std: DataFrame = None) -> list:
    no_normalization = mean is None and std is None
    if no_normalization:
        mean, std = _compute_normalization(dfs[0], num_cols)
    dfs = [df.copy() for df in dfs]
    for df in dfs:
        df[num_cols] = (df[num_cols] - mean) / std
    if no_normalization:
        return dfs, mean, std
    return dfs


def _generate_codes(dfs: List, cat_cols: List) -> dict:
    # combine all dfs together so categories are the same
    tmp = pd.concat([df.copy() for df in dfs], keys=range(len(dfs)))
    for col in cat_cols:
        tmp[col] = tmp[col].astype("category").cat.as_ordered()

    # list of categories for each column (always a column for None)
    codes = {col: [None] + list(tmp[col].cat.categories) for col in cat_cols}

    return codes


def _categorize(dfs: List, cat_cols: List, codes: Dict = None) -> list:
    # combine all dfs together so categories are the same
    tmp = pd.concat([df.copy() for df in dfs], keys=range(len(dfs)))
    for col in cat_cols:
        tmp[col] = tmp[col].astype("category").cat.as_ordered()

    no_codes = codes is None
    if no_codes:
        codes = {col: [None] + list(tmp[col].cat.categories) for col in cat_cols}

    # apply codes to each column
    tmp[cat_cols] = tmp[cat_cols].apply(lambda x: x.cat.codes)

    # we add one here as Nones are -1, so they turn into 0's
    tmp[cat_cols] = tmp[cat_cols] + 1

    # split dfs
    dfs = [tmp.xs(i) for i in range(len(dfs))]
    if no_codes:
        return dfs, codes
    return dfs


def _pre_transform(
    dfs: List,
    num_cols: List[str],
    cat_cols: List[str],
    codes: Dict,
    mean: DataFrame,
    std: DataFrame,
    target: str = None,
    target_codes: Dict = None,
):
    dfs = _impute(dfs, num_cols)
    dfs = _normalize(dfs, num_cols, mean=mean, std=std)
    dfs = _categorize(dfs, cat_cols, codes=codes)
    if target_codes and target:
        dfs = _categorize(dfs, [target], codes=target_codes)
    return dfs


def _to_cat_vars_numpy(df, cat_cols: List[str]) -> list:
    if isinstance(df, list) and len(df) == 1:
        df = df[0]
    return [c.to_numpy().astype(np.int64) for n, c in df[cat_cols].items()]


def _to_num_vars_numpy(df, num_cols: List[str]) -> list:
    if isinstance(df, list) and len(df) == 1:
        df = df[0]
    return [c.to_numpy().astype(np.float32) for n, c in df[num_cols].items()]
