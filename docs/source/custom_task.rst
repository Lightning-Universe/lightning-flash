Tutorial: Creating a Custom Task
================================

In this tutorial we will go over the process of creating a custom task,
along with a custom data module.

.. testcode:: python

    import flash

    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from torch import nn
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

The Task: Linear regression
---------------------------

Here we create a basic linear regression task by subclassing
``flash.Task``. For the majority of tasks, you will likely only need to
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

The Data
--------

For a task you will likely need a specific way of loading data. For this
example, lets say we want a ``flash.DataModule`` to be used explicitly
for the prediction of diabetes disease progression. We can create this
``DataModule`` below, wrapping the scikit-learn `Diabetes
dataset <https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset>`__.

.. testcode::

    class DiabetesPipeline(flash.core.data.TaskDataPipeline):
        def after_uncollate(self, samples):
            return [f"disease progression: {float(s):.2f}" for s in samples]

    class DiabetesData(flash.DataModule):
        def __init__(self, batch_size=64, num_workers=0):
            x, y = datasets.load_diabetes(return_X_y=True)
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float().unsqueeze(1)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=0)

            train_ds = TensorDataset(x_train, y_train)
            test_ds = TensorDataset(x_test, y_test)

            super().__init__(
                train_ds=train_ds,
                test_ds=test_ds,
                batch_size=batch_size,
                num_workers=num_workers
            )
            self.num_inputs = x.shape[1]

        @staticmethod
        def default_pipeline():
            return DiabetesPipeline()

You’ll notice we added a ``DataPipeline``, which will be used when we
call ``.predict()`` on our model. In this case we want to nicely format
our ouput from the model with the string ``"disease progression"``, but
you could do any sort of post processing you want (see :ref:`datapipeline`).

Fit
---

Like any Flash Task, we can fit our model using the ``flash.Trainer`` by
supplying the task itself, and the associated data:

.. code:: python

    data = DiabetesData()
    model = LinearRegression(num_inputs=data.num_inputs)

    trainer = flash.Trainer(max_epochs=1000)
    trainer.fit(model, data)

With a trained model we can now perform inference. Here we will use a
few examples from the test set of our data:

.. code:: python

    predict_data = torch.tensor([
        [ 0.0199,  0.0507,  0.1048,  0.0701, -0.0360, -0.0267, -0.0250, -0.0026, 0.0037,  0.0403],
        [-0.0128, -0.0446,  0.0606,  0.0529,  0.0480,  0.0294, -0.0176,  0.0343, 0.0702,  0.0072],
        [ 0.0381,  0.0507,  0.0089,  0.0425, -0.0428, -0.0210, -0.0397, -0.0026, -0.0181,  0.0072],
        [-0.0128, -0.0446, -0.0235, -0.0401, -0.0167,  0.0046, -0.0176, -0.0026, -0.0385, -0.0384],
        [-0.0237, -0.0446,  0.0455,  0.0907, -0.0181, -0.0354,  0.0707, -0.0395, -0.0345, -0.0094]])

    model.predict(predict_data)

Because of our custom data pipeline’s ``after_uncollate`` method, we
will get a nicely formatted output like the following:

.. code::

   ['disease progression: 155.90',
    'disease progression: 156.59',
    'disease progression: 152.69',
    'disease progression: 149.05',
    'disease progression: 150.90']
