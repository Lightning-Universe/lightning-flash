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
from typing import Any, Dict, List, Optional, Type, Union

from torch.utils.data.sampler import Sampler

from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.io.input import Input
from flash.core.data.io.input_transform import INPUT_TRANSFORM_TYPE, InputTransform
from flash.core.utilities.imports import _PANDAS_AVAILABLE, _TABULAR_TESTING
from flash.core.utilities.stages import RunningStage
from flash.tabular.forecasting.input import TabularForecastingDataFrameInput

if _PANDAS_AVAILABLE:
    from pandas.core.frame import DataFrame
else:
    DataFrame = object


# Skip doctests if requirements aren't available
if not _TABULAR_TESTING:
    __doctest_skip__ = ["TabularForecastingData", "TabularForecastingData.*"]


class TabularForecastingData(DataModule):
    """The ``TabularForecastingData`` class is a :class:`~flash.core.data.data_module.DataModule` with a set of
    classmethods for loading data for tabular forecasting."""

    input_transform_cls = InputTransform

    @property
    def parameters(self) -> Optional[Dict[str, Any]]:
        """The ``parameters`` dictionary from the ``TimeSeriesDataSet`` object created from the train data when
        constructing the ``TabularForecastingData`` object."""
        return getattr(self.train_dataset, "parameters", None)

    @classmethod
    def from_data_frame(
        cls,
        time_idx: Optional[str] = None,
        target: Optional[Union[str, List[str]]] = None,
        group_ids: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        train_data_frame: Optional[DataFrame] = None,
        val_data_frame: Optional[DataFrame] = None,
        test_data_frame: Optional[DataFrame] = None,
        predict_data_frame: Optional[DataFrame] = None,
        input_cls: Type[Input] = TabularForecastingDataFrameInput,
        transform: INPUT_TRANSFORM_TYPE = InputTransform,
        transform_kwargs: Optional[Dict] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        val_split: Optional[float] = None,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        sampler: Optional[Type[Sampler]] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **input_kwargs: Any,
    ) -> "TabularForecastingData":
        """Creates a :class:`~flash.tabular.forecasting.data.TabularForecastingData` object from the given data
        frames.

        .. note::

            The ``time_idx``, ``target``, and ``group_ids`` do not need to be provided if ``parameters`` are passed
            instead. These can be obtained from the
            :attr:`~flash.tabular.forecasting.data.TabularForecastingData.parameters` attribute of the
            :class:`~flash.tabular.forecasting.data.TabularForecastingData` object that contains your training data.

        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            time_idx: Column denoting the time index of each observation.
            target: Column denoting the target or list of columns denoting the target.
            group_ids: List of column names identifying a time series. This means that the group_ids identify a sample
                together with the time_idx. If you have only one timeseries, set this to the name of a column that is
                constant.
            parameters: Parameters to use for the timeseries if ``time_idx``, ``target``, and ``group_ids`` are not
                provided (e.g. when loading data for inference or validation).
            train_data_frame: The pandas DataFrame to use when training.
            val_data_frame: The pandas DataFrame to use when validating.
            test_data_frame: The pandas DataFrame to use when testing.
            predict_data_frame: The pandas DataFrame to use when predicting.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            input_kwargs: Additional keyword arguments to be used when creating the TimeSeriesDataset.

        Returns:
            The constructed :class:`~flash.tabular.forecasting.data.TabularForecastingData`.

        Examples
        ________

        .. testsetup::

            >>> from pytorch_forecasting.data.examples import generate_ar_data
            >>> data = generate_ar_data(seasonality=10.0, timesteps=100, n_series=5, seed=42)

        We have a DataFrame `data` with the following contents:

        .. doctest::

            >>> data.head(3)
               series  time_idx     value
            0       0         0 -0.000000
            1       0         1  0.141552
            2       0         2  0.232782

        .. doctest::

            >>> from pandas import DataFrame
            >>> from flash import Trainer
            >>> from flash.tabular import TabularForecaster, TabularForecastingData
            >>> datamodule = TabularForecastingData.from_data_frame(
            ...     "time_idx",
            ...     "value",
            ...     ["series"],
            ...     train_data_frame=data,
            ...     predict_data_frame=DataFrame.from_dict(
            ...         {
            ...             "time_idx": list(range(50)),
            ...             "value": [0.0] * 50,
            ...             "series": [0] * 50,
            ...         }
            ...     ),
            ...     time_varying_unknown_reals=["value"],
            ...     max_encoder_length=30,
            ...     max_prediction_length=20,
            ...     batch_size=32,
            ... )
            >>> model = TabularForecaster(
            ...     datamodule.parameters,
            ...     backbone="n_beats",
            ...     backbone_kwargs={"widths": [16, 256]},
            ... )
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        .. testcleanup::

            >>> del data
        """

        ds_kw = dict(
            time_idx=time_idx,
            group_ids=group_ids,
            target=target,
            parameters=parameters,
            **input_kwargs,
        )

        train_input = input_cls(RunningStage.TRAINING, train_data_frame, **ds_kw)
        ds_kw["parameters"] = train_input.parameters if train_input else parameters

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_data_frame, **ds_kw),
            input_cls(RunningStage.TESTING, test_data_frame, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data_frame, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            data_fetcher=data_fetcher,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
