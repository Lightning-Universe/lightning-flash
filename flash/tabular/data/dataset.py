from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from torch.utils.data import Dataset


def _impute(dfs: List, num_cols: List):
    dfs = [df.copy() for df in dfs]
    for col in num_cols:
        for df in dfs:
            df[col] = df[col].fillna(dfs[0][col].median())
    return dfs


def _compute_normalization(df: DataFrame, num_cols: List) -> Tuple:
    return df[num_cols].mean(), df[num_cols].std()


def _normalize(dfs: List[DataFrame], num_cols: List, mean: DataFrame = None, std: DataFrame = None):
    no_normalization = mean is None and std is None
    if no_normalization:
        mean, std = _compute_normalization(dfs[0], num_cols)
    dfs = [df.copy() for df in dfs]
    for df in dfs:
        df[num_cols] = (df[num_cols] - mean) / std
    if no_normalization:
        return dfs, mean, std
    return dfs


def _generate_codes(dfs: List, cat_cols: List):
    # combine all dfs together so categories are the same
    tmp = pd.concat([df.copy() for df in dfs], keys=range(len(dfs)))
    for col in cat_cols:
        tmp[col] = tmp[col].astype("category").cat.as_ordered()

    # list of categories for each column (always a column for None)
    codes = {col: [None] + list(tmp[col].cat.categories) for col in cat_cols}

    return codes


def _categorize(dfs: List, cat_cols: List, codes: Dict = None):
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


def _pre_transform(dfs: List, num_cols: List, cat_cols: List, codes: Dict, mean: DataFrame, std: DataFrame):
    dfs = _impute(dfs, num_cols)
    dfs = _normalize(dfs, num_cols, mean=mean, std=std)
    dfs = _categorize(dfs, cat_cols, codes=codes)
    return dfs


def _to_cat_vars_numpy(df, cat_cols):
    if isinstance(df, list) and len(df) == 1:
        df = df[0]
    return [c.to_numpy().astype(np.int64) for n, c in df[cat_cols].items()]


def _to_num_cols_numpy(df, num_cols):
    if isinstance(df, list) and len(df) == 1:
        df = df[0]
    return [c.to_numpy().astype(np.float32) for n, c in df[num_cols].items()]


def _dfs_to_samples(dfs, cat_cols, num_cols, regression=False):
    num_samples = sum([len(df) for df in dfs])
    cat_vars_list = []
    num_vars_list = []
    for df in dfs:
        cat_vars = _to_cat_vars_numpy(df, cat_cols)
        num_vars = _to_num_cols_numpy(df, num_cols)
        cat_vars_list.append(cat_vars)
        cat_vars_list.append(num_vars_list)

    cat_vars = np.stack(cat_vars, 1) if len(cat_vars) else np.zeros((num_samples, 0))
    num_vars = np.stack(num_vars, 1) if len(num_vars) else np.zeros((num_samples, 0))
    return [(c, n) for c, n in zip(cat_vars, num_vars)]


class PandasDataset(Dataset):

    def __init__(self, df, cat_cols, num_cols, target_col, regression=False, predict=False):
        self._num_samples = len(df)
        self.predict = predict
        cat_vars = _to_cat_vars_numpy(df, cat_cols)
        num_vars = _to_num_cols_numpy(df, num_cols)

        if not predict:
            self.target = df[target_col].to_numpy().astype(np.float32 if regression else np.int64)

        self.cat_vars = np.stack(cat_vars, 1) if len(cat_vars) else np.zeros((len(self), 0))
        self.num_vars = np.stack(num_vars, 1) if len(num_vars) else np.zeros((len(self), 0))

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        target = -1 if self.predict else self.target[idx]
        return (self.cat_vars[idx], self.num_vars[idx]), target
