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
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset

from flash.core.data.utils import download_data
from flash.core.utilities.imports import _PANDAS_AVAILABLE, _SKLEARN_AVAILABLE, _TABULAR_AVAILABLE

if _PANDAS_AVAILABLE:
    import pandas as pd
    from pandas.core.frame import DataFrame
else:
    DataFrame = None

if _SKLEARN_AVAILABLE:
    from sklearn.model_selection import train_test_split


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


def _dfs_to_samples(dfs, cat_cols: List[str], num_cols: List[str]) -> list:
    num_samples = sum([len(df) for df in dfs])
    cat_vars_list = []
    num_vars_list = []
    for df in dfs:
        cat_vars = _to_cat_vars_numpy(df, cat_cols)
        num_vars = _to_num_vars_numpy(df, num_cols)
        cat_vars_list.append(cat_vars)
        cat_vars_list.append(num_vars_list)

    # todo: assumes that dfs is not empty
    cat_vars = np.stack(cat_vars, 1) if len(cat_vars) else np.zeros((num_samples, 0))
    num_vars = np.stack(num_vars, 1) if len(num_vars) else np.zeros((num_samples, 0))
    return list(zip(cat_vars, num_vars))


class PandasDataset(Dataset):

    def __init__(
        self,
        df: DataFrame,
        cat_cols: List[str],
        num_cols: List[str],
        target_col: str,
        is_regression: bool = False,
        predict: bool = False
    ):
        self._num_samples = len(df)
        self.predict = predict
        cat_vars = _to_cat_vars_numpy(df, cat_cols)
        num_vars = _to_num_vars_numpy(df, num_cols)

        if not predict:
            self.target = df[target_col].to_numpy().astype(np.float32 if is_regression else np.int64)

        self.cat_vars = np.stack(cat_vars, 1) if len(cat_vars) else np.zeros((len(self), 0))
        self.num_vars = np.stack(num_vars, 1) if len(num_vars) else np.zeros((len(self), 0))

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx):
        target = -1 if self.predict else self.target[idx]
        return (self.cat_vars[idx], self.num_vars[idx]), target


def titanic_data_download(path: str, predict_size: float = 0.1) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

    path_data = os.path.join(path, "titanic.csv")
    download_data("https://pl-flash-data.s3.amazonaws.com/titanic.csv", path_data)

    if set(os.listdir(path)) != {"predict.csv", "titanic.csv"}:
        assert 0 < predict_size < 1
        df = pd.read_csv(path_data)
        df_train, df_predict = train_test_split(df, test_size=predict_size)
        df_train.to_csv(path_data)
        df_predict = df_predict.drop(columns=["Survived"])
        df_predict.to_csv(os.path.join(path, "predict.csv"))
