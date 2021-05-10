***********
Quick Start
***********

Flash is a high-level deep learning framework for fast prototyping, baselining, finetuning and solving deep learning problems. It features a set of tasks for you to use for inference and finetuning out of the box, and an easy to implement API to customize every step of the process for full flexibility.

Flash is built for beginners with a simple API that requires very little deep learning background, and for data scientists, kagglers, applied ML practitioners and deep learning researchers that want a quick way to get a deep learning baseline with advnaced features `Pytorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_ offers.


Why Flash?
==========

For getting started with Deep Learning
--------------------------------------

Easy to learn
^^^^^^^^^^^^^
If you are just getting started with deep learning, Flash offers common deep learning tasks you can use out-of-the-box in a few lines of code, no math, fancy nn.Modules or research experience required!

Easy to scale
^^^^^^^^^^^^^
Flash is built on top of `Pytorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_,
a powerful deep learning research framework for training models at scale. With the power of Lightning,
you can train your flash tasks on any hardware: CPUs, GPUs or TPUs without any code changes.

Easy to upskill
^^^^^^^^^^^^^^^
If you want create more complex and custmoized models, you can refactor any part of flash with PyTorch or `Pytorch Lightning
<https://github.com/PyTorchLightning/pytorch-lightning>`_ components to get all the flexibility you need. Lightning is just
organized PyTorch with the unecessary engineering details abstracted away.

- Flash (high level)
- Lightning (mid-level)
- PyTorch (low-level)

When you need more flexibility you can build your own tasks or simply use Lightning directly.

For Deep learning research
--------------------------

Quickest way to a baseline
^^^^^^^^^^^^^^^^^^^^^^^^^^
`Pytorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_ is designed to abstract away unnecessary boilerplate, while enabling maximal flexibility. In order to provide full flexibility, solving very common deep learning problems such as classification in Lightning still requires some boilerplate. It can still take quite some time to get a baseline model running on a new dataset or out of domain task. We created Flash to answer our users need for a super quick way to baseline for Lightning using proven backbones for common data patterns. Flash aims to be the easiest starting point for your research- start with a Flash Task to benchmark against, and override any part of flash with Lightning or PyTorch components on your way to SOTA research.

Flexibility where you want it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Flash tasks are essentialy LightningModules, and the Flash Trainer is a thin wrapper for the Lightning Trainer. You can use your own LightningModule instead of the Flash task, the Lightning Trainer instead of the flash trainer, etc. Flash helps you focus even more only on your research, and less on anything else.

Standard best practices
^^^^^^^^^^^^^^^^^^^^^^^
Flash tasks implement the standard best practices for a variety of diffrent models and domains, to save you time digging through different implementations. Flash abstracts even more details than lightning, allowing deep learning experts to share their tips and tricks for solving scoped deep learning problems.

.. tip::

    Read :doc:`here <reference/flash_to_pl>` to understand when to use Flash vs Lightning.

----

Install
=======

You can install flash using pip or conda:

.. code-block:: bash

    pip install lightning-flash -U

------

Tasks
=====

Flash is comprised of a collection of Tasks. The Flash tasks are laser-focused objects designed to solve a well-defined type of problem, using state-of-the-art methods.

The Flash tasks contain all the relevant information to solve the task at hand- the number of class labels you want to predict, number of columns in your dataset, as well as details on the model architecture used such as loss function, optimizers, etc.

Here are examples of tasks:

.. testcode::

    from flash.text import TextClassifier
    from flash.vision import ImageClassifier
    from flash.tabular import TabularClassifier

.. note:: Tasks are inflexible by definition! To get more flexibility, you can simply use :class:`~pytorch_lightning.core.lightning.LightningModule` directly or modify and existing task in just a few lines.

------

Inference
=========

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
========

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
    from flash.data.utils import download_data
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
=====

When you have enough data, you're likely better off training from scratch instead of finetuning.
Steps here are similar to finetune:


1. Download and set up your own data (:class:`~torch.utils.data.DataLoader` or `LightningModule <https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html>`_ work).
2. Init your task.
3. Init a :class:`flash.core.trainer.Trainer` (or a `Lightning Trainer <https://pytorch-lightning.readthedocs.io/en/stable/trainer.html>`_).
4. Call :func:`flash.core.trainer.Trainer.fit` with your data set.
5. Use your finetuned model for predictions

-----

A few Built-in Tasks
====================

- :doc:`Generic Flash Task <reference/task>`
- :doc:`ImageClassification <reference/image_classification>`
- :doc:`ImageEmbedder <reference/image_embedder>`
- :doc:`TextClassification <reference/text_classification>`
- :doc:`SummarizationTask <reference/summarization>`
- :doc:`TranslationTask <reference/translation>`
- :doc:`TabularClassification <reference/tabular_classification>`

More tasks coming soon!

Contribute a task
-----------------
The lightning + Flash team is hard at work building more tasks for common deep-learning use cases.
But we're looking for incredible contributors like you to submit new tasks!

Join our `Slack <https://join.slack.com/t/pytorch-lightning/shared_invite/zt-pw5v393p-qRaDgEk24~EjiZNBpSQFgQ>`_ to get help becoming a contributor!
