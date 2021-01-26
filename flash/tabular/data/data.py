from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from flash.core.data.datamodule import DataModule
from flash.tabular.data.dataset import _compute_normalization, _generate_codes, _impute, _pre_transform, PandasDataset


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
        self._test_df = None

        if valid_df is not None:
            dfs.append(valid_df)

        if test_df is not None:
            # save for predict function
            self._test_df = test_df.copy()
            self._test_df.drop(target_col, axis=1)
            dfs.append(test_df)

        # impute missing values
        dfs = _impute(dfs, numerical_cols)

        # compute train dataset stats
        self.mean, self.std = _compute_normalization(dfs[0], numerical_cols)

        self.codes = _generate_codes(dfs, categorical_cols)

        dfs = _pre_transform(dfs, numerical_cols, categorical_cols, self.codes, self.mean, self.std)

        # normalize
        self.cat_cols = categorical_cols
        self.num_cols = numerical_cols

        self._num_classes = len(train_df[target_col].unique())

        train_ds = PandasDataset(dfs[0], categorical_cols, numerical_cols, target_col)
        valid_ds = PandasDataset(dfs[1], categorical_cols, numerical_cols, target_col) if valid_df is not None else None
        test_ds = PandasDataset(dfs[-1], categorical_cols, numerical_cols, target_col) if test_df is not None else None
        super().__init__(train_ds, valid_ds, test_ds, batch_size=batch_size, num_workers=num_workers)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data_config(self):
        return {
            "num_classes": self._num_classes,
            "num_features": len(self.cat_cols) + len(self.num_cols),
            "embedding_sizes": self.emb_sizes,
            "codes": self.codes,
            "mean": self.mean,
            "std": self.std,
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
        }

    @property
    def test_df(self):
        return self._test_df

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
        """Creates a TabularData object from pandas DataFrames.

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
            TabualrData: The constructed data module.

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

    @classmethod
    def from_csv(
        cls,
        train_csv,
        target_col: str,
        categorical_cols: List,
        numerical_cols: List,
        valid_csv=None,
        test_csv=None,
        batch_size: int = 1,
        num_workers: int = None,
        val_size: float = None,
        test_size: float = None,
        **pandas_kwargs,
    ):
        """Creates a TextClassificationData object from pandas DataFrames.

        Args:
            train_csv: train data csv file.
            target_col: The column containing the class id.
            categorical_cols: The list of categorical columns.
            numerical_cols: The list of numerical columns.
            valid_csv: validation data csv file.
            test_csv: test data csv file.
            batch_size: the batchsize to use for parallel loading. Defaults to 64.
            num_workers: The number of workers to use for parallelized loading.
                Defaults to None which equals the number of available CPU threads.
            val_size: float between 0 and 1 to create a validation dataset from train dataset
            test_size: float between 0 and 1 to create a test dataset from train validation

        Returns:
            TabularData: The constructed data module.

        Examples::

            text_data = TabularData.from_files("train.csv", label_field="class", text_field="sentence")
        """
        train_df = pd.read_csv(train_csv, **pandas_kwargs)
        valid_df = pd.read_csv(valid_csv, **pandas_kwargs) if valid_csv is not None else None
        test_df = pd.read_csv(test_csv, **pandas_kwargs) if test_csv is not None else None
        return cls.from_df(
            train_df, target_col, categorical_cols, numerical_cols, valid_df, test_df, batch_size, num_workers,
            val_size, test_size
        )

    @property
    def emb_sizes(self):
        """Recommended embedding sizes."""

        # https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
        # The following "formula" provides a general rule of thumb about the number of embedding dimensions:
        # embedding_dimensions =  number_of_categories**0.25

        num_classes = [len(self.codes[cat]) for cat in self.cat_cols]
        emb_dims = [max(int(n**0.25), 16) for n in num_classes]
        return list(zip(num_classes, emb_dims))
