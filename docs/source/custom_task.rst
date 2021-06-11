Tutorial: Creating a Custom Task
================================

In this tutorial we will go over the process of creating a custom :class:`~flash.core.model.Task`,
along with a custom :class:`~flash.core.data.data_module.DataModule`.

.. note:: This tutorial is only intended to help you create a small custom task for a personal project. If you want a more detailed guide, have a look at our :ref:`guide on contributing a task to flash. <contributing>`

The tutorial objective is to create a ``RegressionTask`` to learn to predict if someone has ``diabetes`` or not.
We will use ``scikit-learn`` `Diabetes dataset <https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset>`__.
which is stored as numpy arrays.

.. note::

    Find the complete tutorial example at
    `flash_examples/custom_task.py <https://github.com/PyTorchLightning/lightning-flash/blob/master/flash_examples/custom_task.py>`_.


1. Imports
----------

We first import everything we're going to use and set the random seed using :func:`~pytorch_lightning.utilities.seed.seed_everything`.

.. testcode:: custom_task

    from typing import Any, Callable, Dict, List, Optional, Tuple

    import numpy as np
    import torch
    from pytorch_lightning import seed_everything
    from sklearn import datasets
    from torch import nn, Tensor

    import flash
    from flash.core.data.data_source import DataSource, DefaultDataKeys, DefaultDataSources
    from flash.core.data.process import Preprocess
    from flash.core.data.transforms import ApplyToKeys

    # set the random seeds.
    seed_everything(42)

    ND = np.ndarray


2. The Task: Linear regression
-------------------------------

Here we create a basic linear regression task by subclassing :class:`~flash.core.model.Task`. For the majority of tasks,
you will likely need to override the ``__init__``, ``forward``, and the ``{train,val,test,predict}_step`` methods. The
``__init__`` should be overridden to configure the model and any additional arguments to be passed to the base
:class:`~flash.core.model.Task`. ``forward`` may need to be overridden to apply the model forward pass to the inputs.
It's best practice in flash for the data to be provide as a dictionary which maps string keys to their values. The
``{train,val,test,predict}_step`` methods need to be overridden to extract the data from the input dictionary.

.. testcode:: custom_task

    class RegressionTask(flash.Task):

        def __init__(self, num_inputs, learning_rate=0.2, metrics=None):
            # what kind of model do we want?
            model = torch.nn.Linear(num_inputs, 1)

            # what loss function do we want?
            loss_fn = torch.nn.functional.mse_loss

            # what optimizer to do we want?
            optimizer = torch.optim.Adam

            super().__init__(
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                metrics=metrics,
                learning_rate=learning_rate,
            )

        def training_step(self, batch: Any, batch_idx: int) -> Any:
            return super().training_step(
                (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET]),
                batch_idx,
            )

        def validation_step(self, batch: Any, batch_idx: int) -> None:
            return super().validation_step(
                (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET]),
                batch_idx,
            )

        def test_step(self, batch: Any, batch_idx: int) -> None:
            return super().test_step(
                (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET]),
                batch_idx,
            )

        def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
            return super().predict_step(
                batch[DefaultDataKeys.INPUT],
                batch_idx,
                dataloader_idx,
            )

        def forward(self, x):
            # we don't actually need to override this method for this example
            return self.model(x)

.. note::

    Lightning Flash provides registries.
    Registries are Flash internal key-value database to store a mapping between a name and a function.
    In simple words, they are just advanced dictionary storing a function from a key string.
    They are useful to store list of backbones and make them available for a :class:`~flash.core.model.Task`.
    Check out :ref:`registry` to learn more.


Where is the training step?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most models can be trained simply by passing the output of ``forward`` to the supplied ``loss_fn``, and then passing the
resulting loss to the supplied ``optimizer``. If you need a more custom configuration, you can override ``step`` (which
is called for training, validation, and testing) or override ``training_step``, ``validation_step``, and ``test_step``
individually. These methods behave identically to PyTorch Lightning’s
`methods <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#methods>`__.

Here is the pseudo code behind :class:`~flash.core.model.Task` step:

.. code:: python

    def step(self, batch: Any, batch_idx: int) -> Any:
        """
        The training/validation/test step. Override for custom behavior.
        """
        x, y = batch
        y_hat = self(x)
        # compute the logs, loss and metrics as an output dictionary
        ...
        return output


3.a The DataSource API
----------------------

Now that we have defined our ``RegressionTask``, we need to load our data. We will define a custom ``NumpyDataSource``
which extends :class:`~flash.core.data.data_source.DataSource`. The ``NumpyDataSource`` contains a ``load_data`` and
``predict_load_data`` methods which handle the loading of a sequence of dictionaries from the input numpy arrays. When
loading the train data (``if self.training:``), the ``NumpyDataSource`` sets the ``num_inputs`` attribute of the
optional ``dataset`` argument. Any attributes that are set on the optional ``dataset`` argument will also be set on the
generated ``dataset``.

.. testcode:: custom_task

    class NumpyDataSource(DataSource[Tuple[ND, ND]]):

        def load_data(self, data: Tuple[ND, ND], dataset: Optional[Any] = None) -> List[Dict[str, Any]]:
            if self.training:
                dataset.num_inputs = data[0].shape[1]
            return [{DefaultDataKeys.INPUT: x, DefaultDataKeys.TARGET: y} for x, y in zip(*data)]

        def predict_load_data(self, data: ND) -> List[Dict[str, Any]]:
            return [{DefaultDataKeys.INPUT: x} for x in data]


3.b The Preprocess API
----------------------

Now that we have a :class:`~flash.core.data.data_source.DataSource` implementation, we can define our
:class:`~flash.core.data.process.Preprocess`. The :class:`~flash.core.data.process.Preprocess` object provides a series of hooks
that can be overridden with custom data processing logic and to which transforms can be attached.
It allows the user much more granular control over their data processing flow.

.. note::

    Why introduce :class:`~flash.core.data.process.Preprocess` ?

    The :class:`~flash.core.data.process.Preprocess` object reduces the engineering overhead to make inference on raw data or
    to deploy the model in production environnement compared to a traditional
    `Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_.

    You can override ``predict_{hook_name}`` hooks or the ``default_predict_transforms`` to handle data processing logic
    specific for inference.

The recommended way to define a custom :class:`~flash.core.data.process.Preprocess` is as follows:

- Define an ``__init__`` which accepts transform arguments.
- Pass these arguments through to ``super().__init__`` and specify the ``data_sources`` and the ``default_data_source``.
    - ``data_sources`` gives the :class:`~flash.core.data.data_source.DataSource` objects that work with your :class:`~flash.core.data.process.Preprocess` as a mapping from data source name to :class:`~flash.core.data.data_source.DataSource`. The data source name can be any string, but for our purposes we can use ``NUMPY`` from :class:`~flash.core.data.data_source.DefaultDataSources`.
    - ``default_data_source`` is the name of the data source to use by default when predicting.
- Override the ``get_state_dict`` and ``load_state_dict`` methods. These methods are used to save and load your :class:`~flash.core.data.process.Preprocess` from a checkpoint.
- Override the ``{train,val,test,predict}_default_transforms`` methods to specify the default transforms to use in each stage (these will be used if the transforms passed in the ``__init__`` are ``None``).
    - Transforms are given as a mapping from hook name to callable transforms. You should use :class:`~flash.core.data.transforms.ApplyToKeys` to apply each transform only to specific keys in the data dictionary.

.. testcode:: custom_task

    class NumpyPreprocess(Preprocess):

        def __init__(
            self,
            train_transform: Optional[Dict[str, Callable]] = None,
            val_transform: Optional[Dict[str, Callable]] = None,
            test_transform: Optional[Dict[str, Callable]] = None,
            predict_transform: Optional[Dict[str, Callable]] = None,
        ):
            super().__init__(
                train_transform=train_transform,
                val_transform=val_transform,
                test_transform=test_transform,
                predict_transform=predict_transform,
                data_sources={DefaultDataSources.NUMPY: NumpyDataSource()},
                default_data_source=DefaultDataSources.NUMPY,
            )

        @staticmethod
        def to_float(x: Tensor):
            return x.float()

        @staticmethod
        def format_targets(x: Tensor):
            return x.unsqueeze(0)

        @property
        def to_tensor(self) -> Dict[str, Callable]:
            return {
                "to_tensor_transform": nn.Sequential(
                    ApplyToKeys(
                        DefaultDataKeys.INPUT,
                        torch.from_numpy,
                        self.to_float,
                    ),
                    ApplyToKeys(
                        DefaultDataKeys.TARGET,
                        torch.as_tensor,
                        self.to_float,
                        self.format_targets,
                    ),
                ),
            }

        def default_transforms(self) -> Optional[Dict[str, Callable]]:
            return self.to_tensor

        def get_state_dict(self) -> Dict[str, Any]:
            return self.transforms

        @classmethod
        def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
            return cls(*state_dict)


3.c The DataModule API
----------------------

Now that we have a :class:`~flash.core.data.process.Preprocess` which knows about the
:class:`~flash.core.data.data_source.DataSource` objects it supports, we just need to create a
:class:`~flash.core.data.data_module.DataModule` which has a reference to the ``preprocess_cls`` we want it to use. For any
data source whose name is in :class:`~flash.core.data.data_source.DefaultDataSources`, there is a standard
``DataModule.from_*`` method that provides the expected inputs. So in this case, there is the
:meth:`~flash.core.data.data_module.DataModule.from_numpy` that will use our numpy data source.

.. testcode:: custom_task

    class NumpyDataModule(flash.DataModule):

        preprocess_cls = NumpyPreprocess


You now have a new customized Flash Task! Congratulations !

You can fit, finetune, validate and predict directly with those objects.

4. Fitting
----------

For this task, here is how to fit the ``RegressionTask`` Task on ``scikit-learn`` `Diabetes
dataset <https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset>`__.

Like any Flash Task, we can fit our model using the ``flash.Trainer`` by
supplying the task itself, and the associated data:

.. testcode:: custom_task

    x, y = datasets.load_diabetes(return_X_y=True)
    datamodule = NumpyDataModule.from_numpy(x, y)

    model = RegressionTask(num_inputs=datamodule.train_dataset.num_inputs)

    trainer = flash.Trainer(max_epochs=20, progress_bar_refresh_rate=20, checkpoint_callback=False)
    trainer.fit(model, datamodule=datamodule)


.. testoutput:: custom_task
    :hide:

    ...


5. Predicting
-------------

With a trained model we can now perform inference. Here we will use a few examples from the test set of our data:

.. testcode:: custom_task

    predict_data = np.array([
        [ 0.0199,  0.0507,  0.1048,  0.0701, -0.0360, -0.0267, -0.0250, -0.0026,  0.0037,  0.0403],
        [-0.0128, -0.0446,  0.0606,  0.0529,  0.0480,  0.0294, -0.0176,  0.0343,  0.0702,  0.0072],
        [ 0.0381,  0.0507,  0.0089,  0.0425, -0.0428, -0.0210, -0.0397, -0.0026, -0.0181,  0.0072],
        [-0.0128, -0.0446, -0.0235, -0.0401, -0.0167,  0.0046, -0.0176, -0.0026, -0.0385, -0.0384],
        [-0.0237, -0.0446,  0.0455,  0.0907, -0.0181, -0.0354,  0.0707, -0.0395, -0.0345, -0.0094],
    ])

    predictions = model.predict(predict_data)
    print(predictions)

We get the following output:

.. testoutput:: custom_task
    :hide:

    [tensor([...]), tensor([...]), tensor([...]), tensor([...]), tensor([...])]

.. code-block::

    [tensor([189.1198]), tensor([196.0839]), tensor([161.2461]), tensor([130.7591]), tensor([149.1780])]
