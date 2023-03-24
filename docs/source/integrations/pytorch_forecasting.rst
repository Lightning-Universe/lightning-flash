.. _pytorch_forecasting:

###################
PyTorch Forecasting
###################

`PyTorch Forecasting <https://github.com/jdb78/pytorch-forecasting>`__ provides the models and data loading for the :ref:`tabular_forecasting` task in Flash.
As with all of our tasks, you won't typically interact with the components from PyTorch Forecasting directly.
However, `PyTorch Forecasting <https://github.com/jdb78/pytorch-forecasting>`__ provides some built-in plotting and analysis methods that are different for each model which cannot be used directly with the :class:`~flash.tabular.forecasting.model.TabularForecaster`.
Instead, you can access the `PyTorch Forecasting <https://github.com/jdb78/pytorch-forecasting>`__ model object using the :attr:`~flash.tabular.forecasting.model.TabularForecaster.pytorch_forecasting_model` attribute.
In addition, we provide the :func:`~flash.core.integrations.pytorch_forecasting.transforms.convert_predictions` utility to convert predictions from the Flash format into the expected format.
With these, you can train your model and perform inference using Flash but still make use of the plotting and analysis tools built in to `PyTorch Forecasting <https://github.com/jdb78/pytorch-forecasting>`__.

Here's an example, plotting the predictions and interpretation analysis from the NBeats model trained in the :ref:`tabular_forecasting` documentation:

.. literalinclude:: ../../../examples/integrations/pytorch_forecasting/tabular_forecasting_interpretable.py
    :language: python
    :lines: 14-

Here's the visualization:

.. image:: https://pl-flash-data.s3.amazonaws.com/assets/pytorch_forecasting_plot.png
    :width: 100%
