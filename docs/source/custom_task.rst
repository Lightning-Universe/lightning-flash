Tutorial: Creating a Custom Task
================================

In this tutorial we will go over the process of creating a custom task,
along with a custom data module.

1 . Imports
-----------


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

    seed_everything(42)


2 . The Task: Linear regression
-------------------------------

Here we create a basic linear regression task by subclassing
:class:`~flash.Task`. For the majority of tasks, you will likely only need to
override the ``__init__`` and ``forward`` methods.

.. testcode::

    class LinearRegression(flash.Task):

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

Where is the training step?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most models can be trained simply by passing the output of ``forward``
to the supplied ``loss_fn``, and then passing the resulting loss to the
supplied ``optimizer``. If you need a more custom configuration, you can
override ``step`` (which is called for training, validation, and
testing) or override ``training_step``, ``validation_step``, and
``test_step`` individually. These methods behave identically to PyTorch
Lightning’s
`methods <https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html#methods>`__.

3 . The Data
-------------

For a task you will likely need a specific way of loading data.

It is recommended to create a :class:`~flash.data.process.Preprocess` object.
The :class:`~flash.data.process.Preprocess` contains all the processing logic and are similar to ``Callback``.
The user has to override hooks with their processing logic.

.. note::
    As new concepts are being introduced, we strongly encourage the reader to click on :class:`~flash.data.process.Preprocess`
    before going further in the tutorial.

The user would have to implement a :class:`~flash.data.data_module.DataModule` as a way to perform data checks and instantiate the preprocess.

.. note::

   Philosophically, the :class:`~flash.data.process.Preprocess` belongs with the :class:`~flash.data.data_module.DataModule`
   and the :class:`~flash.data.process.Postprocess` with the :class:`~flash.Task`.


3.a The DataModule API
----------------------

First, let's design the user-facing API. The ``NumpyDataModule`` will provide a ``from_xy_dataset`` helper ``classmethod``.

Example::

    x, y = ...
    preprocess_cls = ...
    datamodule = NumpyDataModule.from_xy_dataset(x, y, preprocess_cls)

Here are the `NumpyDataModule`` implementation:

Example::

    from flash import DataModule
    from flash.data.process import Preprocess

    class NumpyDataModule(DataModule):

        @classmethod
        def from_xy_dataset(cls, x: ND, y: ND, preprocess_cls: Preprocess = NumpyPreprocess, batch_size: int = 64, num_workers: int = 0):

            preprocess = preprocess_cls()

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=0)

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

4. Fitting
----------

For this task, we will be using ``scikit-learn`` `Diabetes
dataset <https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset>`__.

Like any Flash Task, we can fit our model using the ``flash.Trainer`` by
supplying the task itself, and the associated data:

.. code:: python

    x, y = datasets.load_diabetes(return_X_y=True)
    datamodule = NumpyDataModule.from_xy_dataset(x, y)
    model = LinearRegression(num_inputs=datamodule.num_inputs)

    trainer = flash.Trainer(max_epochs=1000)
    trainer.fit(model, data)

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
        [-0.0237, -0.0446,  0.0455,  0.0907, -0.0181, -0.0354,  0.0707, -0.0395, -0.0345, -0.0094]])

    predictions = model.predict(predict_data)
    print(predictions)
    #out: [tensor([14.7190]), tensor([14.7100]), tensor([14.7288]), tensor([14.6685]), tensor([14.6687])]


6. Customize PostProcess
------------------------

To customize the postprocessing of this task, you can create a :class:`~flash.data.process.Postprocess` objects and assign it to your model as follows:

.. code:: python

    class CustomPostprocess(Postprocess):

        THRESHOLD = 14.72

        def predict_per_sample_transform(self, pred: Any) -> Any:
            if pred > self.THRESHOLD:

                def send_slack_message(pred):
                    print(f"This prediction: {pred} is above the threshold: {self.THRESHOLD}")

                send_slack_message(pred)
            return pred


    class LinearRegression(flash.Task):

        # ``postprocess_cls`` is a special attribute name used internally
        # to instantiate your Postprocess.
        postprocess_cls = CustomPostprocess

        ...

And when running predict one more time.

.. code:: python

    predict_data = ...

    predictions = model.predict(predict_data)
    # out: This prediction: tensor([14.7288]) is above the threshold: 14.72

    print(predictions)
    # out: [tensor([14.7190]), tensor([14.7100]), tensor([14.7288]), tensor([14.6685]), tensor([14.6687])]
