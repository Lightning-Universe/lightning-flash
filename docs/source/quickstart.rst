Quick Start
===========

Flash is a high-level deep learning framework for fast prototyping, baselining, finetuning and solving deep learning problems. It features a set of tasks for you to use for inference and finetuning out of the box, and an easy to implement API to customize every step of the process.


Flash is excellent for:

- data scientists
- kagglers
- applied corporate researchers
- applied academic researchers


Why Flash?
----------

Flash is built on top of `Pytorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_,
a powerful deep learning research framework for training models at scale. With the power of Lightning,
you can train your flash tasks on any hardware: CPUs, GPUs or TPUs without any code changes.

End-to-end deep learning
^^^^^^^^^^^^^^^^^^^^^^^^
Flash is built for 3 major use-cases:

- Inference (predictions)
- Finetuning
- Training


Scalability
^^^^^^^^^^^
Flash is built on top of `Pytorch Lightning
<https://github.com/PyTorchLightning/pytorch-lightning>`_, a powerful deep learning research framework for training models at scale. With the power of Lightning, you can train your flash tasks on any hardware: CPUs, GPUs or TPUs without any code changes. 

Flexibility
^^^^^^^^^^^
Unlike other high-level frameworks, it's easy to customize the Flash tasks with `Pytorch Lightning
<https://github.com/PyTorchLightning/pytorch-lightning>`_ components to get all the flexibility you need. Lightning is just
organized PyTorch with the unecessary engineering details abstracted away.

- Flash (high level)
- Lightning (mid-level)
- PyTorch (low-level)

When you need more flexibility you can build your own tasks or simply use Lightning directly.

.. tip::

    Read :doc:`here <reference/flash_to_pl>` to understand when to use Flash vs Lightning.

----

Install
-------

You can install flash using pip or conda:

.. code-block:: bash

    pip install lightning-flash -U

------

Tasks
-----
Flash is comprised of a collection of Tasks. The Flash tasks are opinionated and laser-focused objects designed to solve a specific type of problem, using state-of-the-art methods. 

The Flash tasks contain all the relevant information to solve the task at hand- the number of class labels you want to predict, number of columns in your dataset, as well as details on the model architecture used such as loss function, optimizers, etc.

Here are examples of tasks:

.. testcode::

    from flash.text import TextClassifier
    from flash.vision import ImageClassifier
    from flash.tabular import TabularClassifier

.. note:: Tasks are inflexible by definition! To get more flexibility, you can simply use :class:`~pytorch_lightning.core.lightning.LightningModule` directly or modify and existing task in just a few lines.

------

Inference
---------
Inference is the process of generating predictions from trained models. To use a task for inference:

1. Init your task with pretrained weights using a checkpoint (a checkpoint is simply a file that capture the exact value of all parameters used by a model). Local file or URL works.
2. Pass in the data to :func:`flash.core.model.Task.predict`.

|

Here's an example of inference.

.. code-block:: python

    # import our libraries
    from flash.text import TextClassifier

    # 1. Init the finetuned task from URL
    model = TextClassifier.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/text_classification_model.pt")

    # 2. Perform inference from list of sequences
    predictions = model.predict([
        "Turgid dialogue, feeble characterization - Harvey Keitel a judge?.",
        "The worst movie in the history of cinema.",
        "This guy has done a great job with this movie!",
    ])

    # Expect [0,0, 1] which means [negative, negative, positive]
    print(predictions)

-------

Finetune
--------
Finetuning (or transfer-learning) is the process of tweaking a model trained on a large dataset, to your particular (likely much smaller) dataset.
To use a Task for finetuning:

1. Download and set up your own data (:class:`~torch.utils.data.DataLoader` or `LightningModule <https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html>`_ work).
2. Init your task.
3. Init a :class:`flash.core.trainer.Trainer` (or a `Lightning Trainer <https://pytorch-lightning.readthedocs.io/en/stable/trainer.html>`_).
4. Call :func:`flash.core.trainer.Trainer.finetune` with your data set.
5. Use your finetuned model for predictions

|

Here's an example of finetuning.

.. code-block:: python

    import flash
    from flash import download_data
    from flash.vision import ImageClassificationData, ImageClassifier

    # 1. Download the data
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", 'data/')

    # 2. Load the data from folders
    datamodule = ImageClassificationData.from_folders(
        backbone="resnet18",
        train_folder="data/hymenoptera_data/train/",
        valid_folder="data/hymenoptera_data/val/",
        test_folder="data/hymenoptera_data/test/",
    )

    # 3. Build the model using desired Task
    model = ImageClassifier(num_classes=datamodule.num_classes)

    # 4. Create the trainer (run one epoch for demo)
    trainer = flash.Trainer(max_epochs=1)

    # 5. Finetune the model
    trainer.finetune(model, datamodule=datamodule, strategy="freeze")

    # 6. Use the model for predictions
    predictions = model.predict('data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg')
    # Expact 1 -> bee
    print(predictions)

    predictions = model.predict('data/hymenoptera_data/val/ants/2255445811_dabcdf7258.jpg')
    # Expact 0 -> ant
    print(predictions)

    # 7. Save the new model!
    trainer.save_checkpoint("image_classification_model.pt")

Once your model is finetuned, use it for prediction anywhere you want!

.. code-block:: python

    from flash.vision import ImageClassifier

    # load finetuned checkpoint
    model = ImageClassifier.load_from_checkpoint("image_classification_model.pt")

    predictions = model.predict('path/to/your/own/image.png')

----

Train
-----
When you have enough data, you're likely better off training from scratch instead of finetuning.
Steps here are similar to finetune:


1. Download and set up your own data (:class:`~torch.utils.data.DataLoader` or `LightningModule <https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html>`_ work).
2. Init your task.
3. Init a :class:`flash.core.trainer.Trainer` (or a `Lightning Trainer <https://pytorch-lightning.readthedocs.io/en/stable/trainer.html>`_).
4. Call :func:`flash.core.trainer.Trainer.fit` with your data set.
5. Use your finetuned model for predictions

-----

A few Built-in Tasks
--------------------

- :doc:`Task <reference/task>`
- :doc:`ImageClassification <reference/image_classification>`
- :doc:`TextClassification <reference/text_classification>`
- :doc:`TabularClassification <reference/tabular_classification>`

-----

Contribute a task
-----------------
The lightning + Flash team is hard at work building more tasks for common deep-learning use cases.
But we're looking for incredible contributors like you to submit new tasks!

Join our `Slack <https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A>`_ to get help becoming a contributor!
