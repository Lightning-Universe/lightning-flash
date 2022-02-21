.. customcarditem::
   :header: Tabular Forecasting
   :card_description: Learn how to perform time series forecasting with Flash and train an NBeats model on some synthetic data.
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Tabular,Forecasting,Timeseries

.. _tabular_forecasting:

###################
Tabular Forecasting
###################

********
The Task
********

Tabular (or timeseries) forecasting is the task of using historical data to predict future trends in a time varying quantity such as: stock prices, temperature, etc.
The :class:`~flash.tabular.forecasting.model.TabularForecaster` and :class:`~flash.tabular.forecasting.data.TabularForecastingData` enable timeseries forecasting in Flash using `PyTorch Forecasting <https://github.com/jdb78/pytorch-forecasting>`__.

------

*******
Example
*******

Let's look at training the NBeats model on some synthetic data with seasonal changes.
The data could represent many naturally occurring timeseries such as energy demand which fluctuates throughout the day but is also expected to change with the season.
This example is a reimplementation of the `NBeats tutorial from the PyTorch Forecasting docs <https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/ar.html>`__ in Flash.
The NBeats model takes no additional inputs unlike other more complex models such as the `Temporal Fusion Transformer <https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/stallion.html>`__.

Once we've created, we can create the :class:`~flash.tabular.classification.data.TabularData` from our DataFrame using the :func:`~flash.tabular.forecasting.data.TabularForecastingData.from_data_frame` method.
To this method, we provide any configuration arguments that should be used when internally constructing the `TimeSeriesDataSet <https://pytorch-forecasting.readthedocs.io/en/latest/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html>`__.

Next, we create the :class:`~flash.tabular.forecasting.model.TabularForecaster` and train on the data.
We then use the trained :class:`~flash.tabular.forecasting.model.TabularForecaster` for inference.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/tabular_forecasting.py
    :language: python
    :lines: 14-

To learn how to the available backbones / heads for this task, see :ref:`backbones_heads`.

.. note::

    Read more about :ref:`our integration with PyTorch Forecasting <pytorch_forecasting>` to see how to use your Flash model with their built-in plotting capabilities.
