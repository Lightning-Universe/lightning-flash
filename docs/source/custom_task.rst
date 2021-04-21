Tutorial: Creating a Custom Task
================================

In this tutorial we will go over the process of creating a custom :class:`~flash.core.model.Task`,
along with a custom :class:`~flash.data.data_module.DataModule`.


The tutorial objective is to create a ``RegressionTask`` to learn to predict if someone has ``diabetes`` or not.
We will use ``scikit-learn`` `Diabetes dataset <https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset>`__.
which is stored as numpy arrays.

.. note::

    Find the complete tutorial example at
    `flash_examples/custom_task.py <https://github.com/PyTorchLightning/lightning-flash/blob/revamp_doc/flash_examples/custom_task.py>`_.


1. Imports
----------


.. testcode:: python

    from typing import Any, List, Tuple

    import numpy as np
    import torch
    from pytorch_lightning import seed_everything
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from torch import nn

    import flash
    from flash.data.auto_dataset import AutoDataset
    from flash.data.process import Postprocess, Preprocess

    # set the random seeds.
    seed_everything(42)


2. The Task: Linear regression
-------------------------------

Here we create a basic linear regression task by subclassing
:class:`~flash.core.model.Task`. For the majority of tasks, you will likely only need to
override the ``__init__`` and ``forward`` methods.

.. testcode::

    class RegressionTask(flash.Task):

        def __init__(self, num_inputs, learning_rate=0.001, metrics=None):
            # what kind of model do we want?
            model = nn.Linear(num_inputs, 1)

            # what loss function do we want?
            loss_fn = torch.nn.functional.mse_loss

            # what optimizer to do we want?
            optimizer = torch.optim.SGD

            super().__init__(
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                metrics=metrics,
                learning_rate=learning_rate,
            )

        def forward(self, x):
            # we don't actually need to override this method for this example
            return self.model(x)

.. note::

    Lightning Flash provides registries.
    Registries are Flash internal key-value database to store a mapping between a name and a function.
    In simple words, they are just advanced dictionary storing a function from a key string.
    They are useful to store list of backbones and make them available for a :class:`~flash.core.model.Task`.
    Check out to learn more :ref:`registry`.


Where is the training step?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most models can be trained simply by passing the output of ``forward``
to the supplied ``loss_fn``, and then passing the resulting loss to the
supplied ``optimizer``. If you need a more custom configuration, you can
override ``step`` (which is called for training, validation, and
testing) or override ``training_step``, ``validation_step``, and
``test_step`` individually. These methods behave identically to PyTorch
Lightning’s
`methods <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#methods>`__.

Here is the pseudo code behind :class:`~flash.core.model.Task` step.

Example::

    def step(self, batch: Any, batch_idx: int) -> Any:
        """
        The training/validation/test step. Override for custom behavior.
        """
        x, y = batch
        y_hat = self(x)
        # compute the logs, loss and metrics as an output dictionary
        ...
        return output


3.a The DataModule API
----------------------

Now that we have defined our ``RegressionTask``, we need to load our data.
We will define a custom ``NumpyDataModule`` class subclassing :class:`~flash.data.data_module.DataModule`.
This ``NumpyDataModule`` class will provide a ``from_xy_dataset`` helper ``classmethod`` to instantiate
:class:`~flash.data.data_module.DataModule` from x, y numpy arrays.

Here is how it would look like:

Example::

    x, y = ...
    preprocess_cls = ...
    datamodule = NumpyDataModule.from_xy_dataset(x, y, preprocess_cls)

Here is the ``NumpyDataModule`` implementation:

Example::

    from flash import DataModule
    from flash.data.process import Preprocess
    import numpy as np

    ND = np.ndarray

    class NumpyDataModule(DataModule):

        @classmethod
        def from_xy_dataset(
            cls,
            x: ND,
            y: ND,
            preprocess_cls: Preprocess = NumpyPreprocess,
            batch_size: int = 64,
            num_workers: int = 0
        ):

            preprocess = preprocess_cls()

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=.20, random_state=0)

            # Make sure to call ``from_load_data_inputs``.
            # The ``train_load_data_input`` value will be given to ``Preprocess``
            # ``train_load_data`` function.
            dm = cls.from_load_data_inputs(
                train_load_data_input=(x_train, y_train),
                test_load_data_input=(x_test, y_test),
                preprocess=preprocess,  # DON'T FORGET TO PROVIDE THE PREPROCESS
                batch_size=batch_size,
                num_workers=num_workers
            )
            # Some metatada can be accessed from ``train_ds`` directly.
            dm.num_inputs = dm.train_dataset.num_inputs
            return dm


.. note::

    The :class:`~flash.data.data_module.DataModule` provides a ``from_load_data_inputs`` helper function. This function will take care
    of connecting the provided :class:`~flash.data.process.Preprocess` with the :class:`~flash.data.data_module.DataModule`.
    Make sure to instantiate your :class:`~flash.data.data_module.DataModule` with this helper if you rely on :class:`~flash.data.process.Preprocess`
    objects.

3.b The Preprocess API
----------------------

A :class:`~flash.data.process.Preprocess` object provides a series of hooks that can be overridden with custom data processing logic.
It allows the user much more granular control over their data processing flow.

.. note::

    Why introducing :class:`~flash.data.process.Preprocess` ?

    The :class:`~flash.data.process.Preprocess` object reduces the engineering overhead to make inference on raw data or
    to deploy the model in production environnement compared to traditional
    `Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_.

    You can override ``predict_{hook_name}`` hooks to handle data processing logic specific for inference.

Example::

    import torch
    from torch import Tensor
    import numpy as np

    ND = np.ndarray

    class NumpyPreprocess(Preprocess):

        def load_data(self, data: Tuple[ND, ND], dataset: AutoDataset) -> List[Tuple[ND, float]]:
            if self.training:
                dataset.num_inputs = data[0].shape[1]
            return [(x, y) for x, y in zip(*data)]

        def to_tensor_transform(self, sample: Any) -> Tuple[Tensor, Tensor]:
            x, y = sample
            x = torch.from_numpy(x).float()
            y = torch.tensor(y, dtype=torch.float)
            return x, y

        def predict_load_data(self, data: ND) -> ND:
            return data

        def predict_to_tensor_transform(self, sample: ND) -> ND:
            return torch.from_numpy(sample).float()


You now have a new customized Flash Task! Congratulations !

You can fit, finetune, validate and predict directly with those objects.

4. Fitting
----------

For this task, here is how to fit the ``RegressionTask`` Task on ``scikit-learn`` `Diabetes
dataset <https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset>`__.

Like any Flash Task, we can fit our model using the ``flash.Trainer`` by
supplying the task itself, and the associated data:

.. code:: python

    x, y = datasets.load_diabetes(return_X_y=True)
    datamodule = NumpyDataModule.from_xy_dataset(x, y)
    model = RegressionTask(num_inputs=datamodule.num_inputs)

    trainer = flash.Trainer(max_epochs=1000)
    trainer.fit(model, datamodule=datamodule)


5. Predicting
-------------

With a trained model we can now perform inference. Here we will use a
few examples from the test set of our data:

.. code:: python

    predict_data = torch.tensor([
        [ 0.0199,  0.0507,  0.1048,  0.0701, -0.0360, -0.0267, -0.0250, -0.0026, 0.0037,  0.0403],
        [-0.0128, -0.0446,  0.0606,  0.0529,  0.0480,  0.0294, -0.0176,  0.0343, 0.0702,  0.0072],
        [ 0.0381,  0.0507,  0.0089,  0.0425, -0.0428, -0.0210, -0.0397, -0.0026, -0.0181,  0.0072],
        [-0.0128, -0.0446, -0.0235, -0.0401, -0.0167,  0.0046, -0.0176, -0.0026, -0.0385, -0.0384],
        [-0.0237, -0.0446,  0.0455,  0.0907, -0.0181, -0.0354,  0.0707, -0.0395, -0.0345, -0.0094]]
    )

    predictions = model.predict(predict_data)
    print(predictions)
    #out: [tensor([14.7190]), tensor([14.7100]), tensor([14.7288]), tensor([14.6685]), tensor([14.6687])]
