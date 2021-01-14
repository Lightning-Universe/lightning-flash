from typing import List, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from pl_flash.data.datamodule import DataModule
from pl_flash.tabular.data.dataset import PandasDataset


def _categorize(dfs: List, cat_cols: List):

    # combine all dfs together so categories are the same
    tmp = pd.concat([df.copy() for df in dfs], keys=range(len(dfs)))
    for col in cat_cols:
        tmp[col] = tmp[col].astype("category").cat.as_ordered()

    # list of categories for each column (always a column for None)
    codes = {col: [None] + list(tmp[col].cat.categories) for col in cat_cols}

    # apply codes to each column
    tmp[cat_cols] = tmp[cat_cols].apply(lambda x: x.cat.codes)

    # we add one here as Nones are -1, so they turn into 0's
    tmp[cat_cols] = tmp[cat_cols] + 1

    # split dfs
    dfs = [tmp.xs(i) for i in range(len(dfs))]
    return dfs, codes


def _normalize(dfs: List, num_cols: List) -> Tuple:
    dfs = [df.copy() for df in dfs]
    mean, std = dfs[0][num_cols].mean(), dfs[0][num_cols].std()
    for df in dfs:
        df[num_cols] = (df[num_cols] - mean) / std
    return dfs, mean, std


def _impute(dfs: List, num_cols: List):
    dfs = [df.copy() for df in dfs]
    for col in num_cols:
        for df in dfs:
            df[col].fillna(dfs[0][col].median())
    return dfs


class TabularData(DataModule):
    """Data module for tabular tasks"""

    def __init__(
        self,
        train_df,
        categorical_cols: List,
        numerical_cols: List,
        target_col: str,
        valid_df=None,
        test_df=None,
        batch_size=2,
        num_workers=None,
    ):
        dfs = [train_df]

        if valid_df is not None:
            dfs.append(valid_df)

        if test_df is not None:
            dfs.append(test_df)

        dfs, self.codes = _categorize(dfs, categorical_cols)
        dfs = _impute(dfs, numerical_cols)
        dfs, self.mean, self.std = _normalize(dfs, numerical_cols)

        self.cat_cols = categorical_cols
        self.num_cols = numerical_cols

        self._num_classes = len(train_df[target_col].unique())

        train_ds = PandasDataset(dfs[0], categorical_cols, numerical_cols, target_col)
        valid_ds = PandasDataset(dfs[1], categorical_cols, numerical_cols, target_col) if valid_df is not None else None
        test_ds = PandasDataset(dfs[-1], categorical_cols, numerical_cols, target_col) if test_df is not None else None
        super().__init__(train_ds, valid_ds, test_ds, batch_size=batch_size, num_workers=num_workers)

    @property
    def num_features(self):
        return len(self.cat_cols) + len(self.num_cols)

    @property
    def num_classes(self):
        return self._num_classes

    @classmethod
    def from_df(
        cls,
        train_df: pd.DataFrame,
        target_col: str,
        categorical_cols: List,
        numerical_cols: List,
        valid_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
        batch_size: int = 1,
        num_workers: int = None,
        val_size: float = None,
        test_size: float = None,
    ):
        """Creates a TextClassificationData object from pandas DataFrames.

        Args:
            train_df: train data DataFrame
            target_col: The column containing the class id.
            categorical_cols: The list of categorical columns.
            numerical_cols: The list of numerical columns.
            valid_df: validation data DataFrame
            test_df: test data DataFrame
            batch_size: the batchsize to use for parallel loading. Defaults to 64.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads.
            val_size: float between 0 and 1 to create a validation dataset from train dataset
            test_size: float between 0 and 1 to create a test dataset from train validation

        Returns:
            TextClassificationData: The constructed data module.

        Examples::

            text_data = TextClassificationData.from_files("train.csv", label_field="class", text_field="sentence")
        """
        if valid_df is None and isinstance(val_size, float) and isinstance(test_size, float):
            assert 0 < val_size and val_size < 1
            assert 0 < test_size and test_size < 1
            train_df, valid_df = train_test_split(train_df, test_size=(val_size + test_size))

        if test_df is None and isinstance(test_size, float):
            assert 0 < test_size and test_size < 1
            valid_df, test_df = train_test_split(valid_df, test_size=test_size)

        return cls(
            train_df=train_df,
            target_col=target_col,
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols,
            valid_df=valid_df,
            test_df=test_df,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    @property
    def emb_sizes(self):
        """Recommended embedding sizes."""

        # https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
        # The following "formula" provides a general rule of thumb about the number of embedding dimensions:
        # embedding_dimensions =  number_of_categories**0.25

        num_classes = [len(self.codes[cat]) for cat in self.cat_cols]
        emb_dims = [max(int(n ** 0.25), 16) for n in num_classes]
        return list(zip(num_classes, emb_dims))
