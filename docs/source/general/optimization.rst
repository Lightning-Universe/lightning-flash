
.. _optimization:

########################################
Optimization (Optimizers and Schedulers)
########################################

Using optimizers and learning rate schedulers with Flash has become easier and cleaner than ever.

With the use of :ref:`registry`, instantiation of an optimzer or a learning rate scheduler can done with just a string.

Setting an optimizer to a task
==============================

Each task has a built-in method :func:`~flash.core.model.Task.available_optimizers` which will list all the optimizers
registered with Flash.

    >>> from flash.core.classification import ClassificationTask
    >>> ClassificationTask.available_optimizers()
    ['adadelta', ..., 'sgd']

To train / finetune a :class:`~flash.core.model.Task` of your choice, just pass on a string.

.. code-block:: python

    from flash.image import ImageClassifier

    model = ImageClassifier(num_classes=10, backbone="resnet18", optimizer="Adam", learning_rate=1e-4)


In order to customize specific parameters of the Optimizer, pass along a dictionary of kwargs with the string as a tuple.

.. code-block:: python

    from flash.image import ImageClassifier

    model = ImageClassifier(num_classes=10, backbone="resnet18", optimizer=("Adam", {"amsgrad": True}), learning_rate=1e-4)


An alternative to customizing an optimizer using a tuple is to pass it as a callable.

.. code-block:: python

    from functools import partial
    from torch.optim import Adam
    from flash.image import ImageClassifier

    model = ImageClassifier(num_classes=10, backbone="resnet18", optimizer=partial(Adam, amsgrad=True), learning_rate=1e-4)


Setting a Learning Rate Scheduler
=================================

Each task has a built-in method :func:`~flash.core.model.Task.available_lr_schedulers` which will list all the learning
rate schedulers registered with Flash.

    >>> from flash.core.classification import ClassificationTask
    >>> ClassificationTask.available_lr_schedulers()
    ['lambdalr', ..., 'cosineannealingwarmrestarts']

To train / finetune a :class:`~flash.core.model.Task` of your choice, just pass on a string.

.. code-block:: python

    from flash.image import ImageClassifier

    model = ImageClassifier(
        num_classes=10, backbone="resnet18", optimizer="Adam", learning_rate=1e-4, lr_scheduler="constant_schedule"
    )

.. note:: ``"constant_schedule"`` and a few other lr schedulers will be available only if you have installed the ``transformers`` library from Hugging Face.


In order to customize specific parameters of the LR Scheduler, pass along a dictionary of kwargs with the string as a tuple.

.. code-block:: python

    from flash.image import ImageClassifier

    model = ImageClassifier(
        num_classes=10,
        backbone="resnet18",
        optimizer="Adam",
        learning_rate=1e-4,
        lr_scheduler=("StepLR", {"step_size": 10}),
    )


An alternative to customizing the LR Scheduler using a tuple is to pass it as a callable.

.. code-block:: python

    from functools import partial
    from torch.optim.lr_scheduler import CyclicLR
    from flash.image import ImageClassifier

    model = ImageClassifier(
        num_classes=10,
        backbone="resnet18",
        optimizer="Adam",
        learning_rate=1e-4,
        lr_scheduler=partial(CyclicLR, step_size_up=1500, mode="exp_range", gamma=0.5),
    )


Additionally, the ``lr_scheduler`` parameter also accepts the Lightning Scheduler configuration which can be passed on using a tuple.

The Lightning Scheduler configuration is a dictionary which contains the scheduler and its associated configuration. The default configuration is shown below.

.. code-block:: python

    lr_scheduler_config = {
        # REQUIRED: The scheduler instance
        "scheduler": lr_scheduler,
        # The unit of the scheduler's step size, could also be 'step'.
        # 'epoch' updates the scheduler on epoch end whereas 'step'
        # updates it after a optimizer update.
        "interval": "epoch",
        # How many epochs/steps should pass between calls to
        # `scheduler.step()`. 1 corresponds to updating the learning
        # rate after every epoch/step.
        "frequency": 1,
        # Metric to to monitor for schedulers like `ReduceLROnPlateau`
        "monitor": "val_loss",
        # If set to `True`, will enforce that the value specified 'monitor'
        # is available when the scheduler is updated, thus stopping
        # training if not found. If set to `False`, it will only produce a warning
        "strict": True,
        # If using the `LearningRateMonitor` callback to monitor the
        # learning rate progress, this keyword can be used to specify
        # a custom logged name
        "name": None,
    }

When there are schedulers in which the ``.step()`` method is conditioned on a value, such as the ``torch.optim.lr_scheduler.ReduceLROnPlateau`` scheduler,
Flash requires that the Lightning Scheduler configuration contains the keyword ``"monitor"`` set to the metric name that the scheduler should be conditioned on.
Below is an example for this:

.. code-block:: python

    from flash.image import ImageClassifier

    model = ImageClassifier(
        num_classes=10,
        backbone="resnet18",
        optimizer="Adam",
        learning_rate=1e-4,
        lr_scheduler=("reducelronplateau", {"mode": "max"}, {"monitor": "val_accuracy"}),
    )


.. note:: Do not set the ``"scheduler"`` key in the Lightning Scheduler configuration, it will overridden with an instance of the provided scheduler key.


Pre-Registering optimizers and scheduler recipes
================================================

Flash registry also provides the flexiblty of registering functions. This feature is also provided in the Optimizer and Scheduler registry.

Using the ``optimizers`` and ``lr_schedulers`` decorator pertaining to each :class:`~flash.core.model.Task`, custom optimizer and LR scheduler recipes can be pre-registered.

.. code-block:: python

    import torch
    from flash.image import ImageClassifier


    @ImageClassifier.lr_schedulers
    def my_flash_steplr_recipe(optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)


    model = ImageClassifier(backbone="resnet18", num_classes=2, optimizer="Adam", lr_scheduler="my_flash_steplr_recipe")


Provider specific requirements
==============================

Schedulers
**********

Certain LR Schedulers provided by Hugging Face require both ``num_training_steps`` and ``num_warmup_steps``.

In order to use them in Flash, just provide ``num_warmup_steps`` as float between 0 and 1 which indicates the fraction of the training steps
that will be used as warmup steps. Flash's :class:`~flash.core.trainer.Trainer` will take care of computing the number of training steps and
number of warmup steps based on the flags that are set in the Trainer.

.. code-block:: python

    from flash.image import ImageClassifier

    model = ImageClassifier(
        backbone="resnet18",
        num_classes=2,
        optimizer="Adam",
        lr_scheduler=("cosine_schedule_with_warmup", {"num_warmup_steps": 0.1}),
    )
