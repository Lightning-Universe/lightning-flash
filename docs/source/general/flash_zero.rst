.. _flash_zero:

**********
Flash Zero
**********

Flash Zero is a zero-code machine learning platform built directly into lightning-flash.
To get started and view the available tasks, run:

.. code-block:: bash

    flash --help

Customize Trainer and Model arguments
_____________________________________

Flash Zero is built on top of the
`lightning CLI <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html>`_, so the trainer and
model arguments can be configured either from the command line or from a config file.
For example, to run the image classifier for 10 epochs with a `resnet50` backbone you can use:

.. code-block:: bash

    flash image-classification --trainer.max_epochs 10 --model.backbone resnet50

To view all of the available options for a task, run:

.. code-block:: bash

    flash image-classification --help

Using Custom Data
_________________

Flash Zero works with your own data through subcommands. The available subcommands for each task are given at the bottom
of their help pages (e.g. when running :code:`flash image-classification --help`). You can then use the required
subcommand to train on your own data. Let's look at an example using the Hymenoptera data from the
:ref:`image_classification` guide. First download and unzip your data:

.. code-block:: bash

    curl https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip -o hymenoptera_data
    unzip hymenoptera_data.zip

Now train with Flash Zero:

.. code-block:: bash

    flash image-classification from_folders --train_folder ./hymenoptera_data/train

You can view the help page for each subcommand. For example, to view the options for training an image classifier from
folders, you can run:

.. code-block:: bash

    flash image-classification from_folders --help
