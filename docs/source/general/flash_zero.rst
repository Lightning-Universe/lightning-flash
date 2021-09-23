.. _flash_zero:

**********
Flash Zero
**********

Flash Zero is a zero-code machine learning platform.
Here's an image classification example to illustrate with one of the dozens tasks available.

.. code-block:: bash

    flash image_classification from_folders --train_folder ./hymenoptera_data/train


See more tasks by running:

.. code-block:: bash

    flash --help

3 steps to use Flash zero
___________________________

1. Select your task

.. code-block:: bash

    flash image_classification
    # flash text_summarization
    # flast ...

2. Select the models and options for the task

.. code-block:: bash

    flash image_classification --trainer.max_epochs 10 --model.backbone resnet50
    # flash text_summarization --trainer.max_epochs 10 --model.backbone bert
    # flast ...

3. Pass in your own data

.. code-block:: bash

    flash image_classification --trainer.max_epochs 10 --model.backbone resnet50 from_folders --train_folder ./hymenoptera_data/train


Some examples:
______________

Image classification

.. code-block:: bash

    flash image_classification from_folders --train_folder ./hymenoptera_data/train

Other task 2

.. code-block:: bash

    flash task_2 from_folders --train_folder ./hymenoptera_data/train

Other task 3

.. code-block:: bash

    flash task_3 from_folders --train_folder ./hymenoptera_data/train

CLI options
___________


Flash Zero is built on top of the
`lightning CLI <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html>`_, so the trainer and
model arguments can be configured either from the command line or from a config file.
For example, to run the image classifier for 10 epochs with a `resnet50` backbone you can use:

.. code-block:: bash

    flash image_classification --trainer.max_epochs 10 --model.backbone resnet50

To view all of the available options for a task, run:

.. code-block:: bash

    flash image_classification --help

Using Your Own Data
___________________

Flash Zero works with your own data through subcommands. The available subcommands for each task are given at the bottom
of their help pages (e.g. when running :code:`flash image-classification --help`). You can then use the required
subcommand to train on your own data. Let's look at an example using the Hymenoptera data from the
:ref:`image_classification` guide. First, download and unzip your data:

.. code-block:: bash

    curl https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip -o hymenoptera_data
    unzip hymenoptera_data.zip

Now train with Flash Zero:

.. code-block:: bash

    flash image_classification from_folders --train_folder ./hymenoptera_data/train

You can view the help page for each subcommand. For example, to view the options for training an image classifier from
folders, you can run:

.. code-block:: bash

    flash image_classification from_folders --help
