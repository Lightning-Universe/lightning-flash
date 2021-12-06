.. _quick_start:

***********
Quick Start
***********

Flash is a high-level deep learning framework for fast prototyping, baselining, finetuning and solving deep learning problems. It features a set of tasks for you to use for inference and finetuning out of the box, and an easy to implement API to customize every step of the process for full flexibility.

Flash is built for beginners with a simple API that requires very little deep learning background, and for data scientists, Kagglers, applied ML practitioners and deep learning researchers that want a quick way to get a deep learning baseline with advanced features `PyTorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_ offers.


Why Flash?
==========

For getting started with Deep Learning
--------------------------------------

Easy to learn
^^^^^^^^^^^^^
If you are just getting started with deep learning, Flash offers common deep learning tasks you can use out-of-the-box in a few lines of code, no math, fancy nn.Modules or research experience required!

Easy to scale
^^^^^^^^^^^^^
Flash is built on top of `PyTorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_,
a powerful deep learning research framework for training models at scale. With the power of Lightning,
you can train your flash tasks on any hardware: CPUs, GPUs or TPUs without any code changes.

Easy to upskill
^^^^^^^^^^^^^^^
If you want to create more complex and customized models, you can refactor any part of flash with PyTorch or `PyTorch Lightning
<https://github.com/PyTorchLightning/pytorch-lightning>`_ components to get all the flexibility you need. Lightning is just
organized PyTorch with the unnecessary engineering details abstracted away.

- Flash (high-level)
- Lightning (mid-level)
- PyTorch (low-level)

When you need more flexibility you can build your own tasks or simply use Lightning directly.

For Deep learning research
--------------------------

Quickest way to a baseline
^^^^^^^^^^^^^^^^^^^^^^^^^^
`PyTorch Lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_ is designed to abstract away unnecessary boilerplate, while enabling maximal flexibility. In order to provide full flexibility, solving very common deep learning problems such as classification in Lightning still requires some boilerplate. It can still take quite some time to get a baseline model running on a new dataset or out of domain task. We created Flash to answer our users need for a super quick way to baseline for Lightning using proven backbones for common data patterns. Flash aims to be the easiest starting point for your research- start with a Flash Task to benchmark against, and override any part of flash with Lightning or PyTorch components on your way to SOTA research.

Flexibility where you want it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Flash tasks are essentially LightningModules, and the Flash Trainer is a thin wrapper for the Lightning Trainer. You can use your own LightningModule instead of the Flash task, the Lightning Trainer instead of the flash trainer, etc. Flash helps you focus even more only on your research, and less on anything else.

Standard best practices
^^^^^^^^^^^^^^^^^^^^^^^
Flash tasks implement the standard best practices for a variety of different models and domains, to save you time digging through different implementations. Flash abstracts even more details than Lightning, allowing deep learning experts to share their tips and tricks for solving scoped deep learning problems.

------

Tasks
=====

Flash is comprised of a collection of Tasks. The Flash tasks are laser-focused objects designed to solve a well-defined type of problem, using state-of-the-art methods.

The Flash tasks contain all the relevant information to solve the task at hand- the number of class labels you want to predict, number of columns in your dataset, as well as details on the model architecture used such as loss function, optimizers, etc.

Here are examples of tasks:

.. testcode::

    from flash.text import TextClassifier
    from flash.image import ImageClassifier
    from flash.tabular import TabularClassifier

.. note:: Tasks are inflexible by definition! To get more flexibility, you can simply use :class:`~pytorch_lightning.core.lightning.LightningModule` directly or modify an existing task in just a few lines.

------

Inference
=========

Inference is the process of generating predictions from trained models. To use a task for inference:

1. Init your task with pretrained weights using a checkpoint (a checkpoint is simply a file that capture the exact value of all parameters used by a model). Local file or URL works.
2. Pass in the data to :func:`flash.core.model.Task.predict`.

|

Here's an example of inference:

.. testcode::

    # import our libraries
    from flash import Trainer
    from flash.text import TextClassifier, TextClassificationData

    # 1. Init the finetuned task from URL
    model = TextClassifier.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/0.6.0/text_classification_model.pt")

    # 2. Perform inference from list of sequences
    trainer = Trainer()
    datamodule = TextClassificationData.from_lists(
        predict_data=[
            "Turgid dialogue, feeble characterization - Harvey Keitel a judge?.",
            "The worst movie in the history of cinema.",
            "This guy has done a great job with this movie!",
        ]
    )
    print(predictions)

We get the following output:

.. testoutput::
    :hide:

    ...

.. testcode::
    :hide:

    assert all(
        [
            all([prediction in ["positive", "negative"] for prediction in prediction_batch])
            for prediction_batch in predictions
        ]
    )

.. code-block::

    [["negative", "negative", "positive"]]

-------

Finetuning
==========

Finetuning (or transfer-learning) is the process of tweaking a model trained on a large dataset, to your particular (likely much smaller) dataset.
All Flash tasks have pre-trained backbones that are already trained on large datasets such as ImageNet. Finetuning on pretrained models decreases training time significantly.

.. include:: common/finetuning_example.rst

-----

Training
========

When you have enough data, you're likely better off training from scratch instead of finetuning.

.. include:: common/training_example.rst

-----

A few Built-in Tasks
====================

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
