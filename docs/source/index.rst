.. flash documentation master file, created by
   sphinx-quickstart on Sat Sep 19 16:37:02 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pytorch Lightning Flash
=======================

Flash is a lightweight wrapper of `Pytorch Lightning
<https://github.com/PyTorchLightning/pytorch-lightning>`_, designed to make it
super easy to apply common deep learning models to your own data.

We seperate the developement process into 3 easy steps: data, model, and train:

.. code-block:: python

    from flash.vision import ImageClassifier, ImageClassificationData
    from flash.text import TextClassifier, TextClassificationData
    from flash.tabular import TabularClassifier, TabularData
    import pytorch_lightning as pl

    # Step 1: Data
    data = ... # pick from a variety of task specific datasets

    # Step 2: Task
    model = ... # pick our task specific model, built with all of the best practices


    # Step 3: Train
    # Train your model using PyTorchLightning, gaining access to all of its features like
    # multi-GPU training and even TPU support!
    pl.Trainer().fit(model, data)


.. toctree::
   :maxdepth: 2
   :caption: Get started:

   quickstart
   installation
   custom_task

.. toctree::
   :maxdepth: 3
   :caption: API Reference:

   reference/image_classification
   reference/text_classification
   reference/tabular_classification
   reference/model
   reference/data
   reference/training
   reference/predictions


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
