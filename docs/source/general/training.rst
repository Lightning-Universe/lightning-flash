
.. _training:

*********************
Training from scratch
*********************

Some Flash tasks have been pretrained on large data sets.
To accelerate your training, calling the :func:`~flash.core.trainer.Trainer.finetune` method using a pretrained backbone will fine-tune the backbone to generate a model customized to your data set and desired task.

From the :ref:`quick_start` guide.

.. include:: ../common/training_example.rst


Training options
================

Flash tasks supports many advanced training functionalities out-of-the-box, such as:

* limit number of epochs

.. code-block:: python

    # train for 10 epochs
    flash.Trainer(max_epochs=10)

* Training on GPUs

.. code-block:: python

    # train on 1 GPU
    flash.Trainer(gpus=1)

* Training on multiple GPUs

.. code-block:: python

    # train on multiple GPUs
    flash.Trainer(gpus=4)

.. code-block:: python

    # train on gpu 1, 3, 5 (3 gpus total)
    flash.Trainer(gpus=[1, 3, 5])

* Using mixed precision training

.. code-block:: python

    # Multi GPU with mixed precision
    flash.Trainer(gpus=2, precision=16)

* Training on TPUs

.. code-block:: python

    # Train on TPUs
    flash.Trainer(accelerator="tpu", num_devices=8)

You can add to the flash Trainer any argument from the Lightning trainer! Learn more about the Lightning Trainer `here <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_.
