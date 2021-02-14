.. _finetuning:

**********
Finetuning
**********

Finetuning (or transfer-learning) is the process of tweaking a model trained on a large dataset, to your particular (likely much smaller) dataset.

------

Terminology
===========
Here are common terms you need to be familiar with:

.. list-table:: Terminology
   :widths: 20 80
   :header-rows: 1

   * - Term
     - Definition
   * - Finetuning
     - The process of tweaking a model trained on a large dataset, to your particular (likely much smaller) dataset
   * - Transfer learning
     - The common name for finetuning
   * - Backbone
     - The neural network that was pretrained on a different dataset
   * - Head
     - Another neural network (usually smaller) that maps the backbone to your particular dataset
   * - Freeze
     - Disabling gradient updates to a model (ie: not learning)
   * - Unfreeze
     - Enabling gradient updates to a model


------

3 steps to finetune in Flash
============================

All Flash tasks have a pre-trained backbone that was already trained on large datasets such as ImageNet. Finetuning on already pretrained models decrease training time significantly.

To finetune using Flash, follow these 3 steps:

1. Load your data and organize it using a DataModule customized for the task.
2. Pick a Task which has all the state-of-the-art built in (example: :class:`~flash.vision.ImageClassifier`).
3. Choose a Finetune strategy and call the :func:`~flash.core.trainer.Trainer.finetune` method

|

Here are the steps in code

.. code-block:: python

    import flash
    from flash import download_data
    from flash.vision import ImageClassificationData, ImageClassifier

    # 1. download and organize the data
    download_data("https://download.pytorch.org/tutorial/hymenoptera_data.zip", 'data/')

    data = ImageClassificationData.from_folders(
        train_folder="data/hymenoptera_data/train/",
        valid_folder="data/hymenoptera_data/val/"
    )

    # 2. build the task
    task = ImageClassifier(num_classes=2)

    # 3. Build the trainer and finetune! In this case, using the no_freeze strategy
    trainer = flash.Trainer()
    trainer.finetune(task, data, strategy="no_freeze")

.. tip:: If you have a large dataset and prefer to train from scratch, see the :ref:`training` guide.

----

Using a finetuned model
=======================
Once you've finetuned, use the model to predict.

.. code-block:: python

    predictions = task.predict('data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg')
    print(predictions)

Or use a different checkpoint for prediction

.. code-block:: python

    # Save the checkpoint while training.
    trainer.save_checkpoint("image_classification_model.pt")

    # load the finetuned model
    classifier = ImageClassifier.load_from_checkpoint('image_classification_model.pt')

    # predict!
    predictions = classifier.predict('data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg')
    print(predictions)

------

Finetune strategies
===================

Finetuning is very task specific. Each task encodes the best finetuning practices for that task.
However, Flash gives you a few default strategies for finetuning.

Finetuning operates on two things, the model backbone and the head. The backbone
is the neural network that was pre-trained. The head is another neural network that bridges between the backbone
and your particular dataset.

no_freeze
---------
In this strategy, the backbone and the head are unfrozen from the beginning.

.. code-block:: python

    trainer.finetune(task, data, strategy='no_freeze')

In pseudocode, this looks like:

.. code-block:: python

    backbone = Resnet50()
    head = nn.Linear(...)

    backbone.unfreeze()
    head.unfreeze()

    train(backbone, head)

freeze
------
The freeze strategy keeps the backbone frozen throughout.

.. code-block:: python

    trainer.finetune(task, data, strategy='freeze')

The psedocode looks like:

.. code-block:: python

    backbone = Resnet50()
    head = nn.Linear(...)

    # freeze backbone
    backbone.freeze()
    head.unfreeze()

    train(backbone, head)

freeze_unfreeze
---------------
In this strategy, the backbone is frozen for 10 epochs then unfrozen.

.. code-block:: python

    trainer.finetune(model, data, strategy='freeze_unfreeze')

.. code-block:: python

    from flash.core.finetuning import FreezeUnfreeze

    # finetune for 10 epochs. Backbone will be frozen for 5 epochs.
    trainer = flash.Trainer(max_epochs=10)
    trainer.finetune(model, data, strategy=FreezeUnfreeze(unfreeze_epoch=5))

Under the hood, the pseudocode looks like:

.. code-block:: python

    backbone = Resnet50()
    head = nn.Linear(...)

    # freeze backbone
    backbone.freeze()
    head.unfreeze()

    train(backbone, head, epochs=10)

    # unfreeze after 10 epochs
    backbone.unfreeze()

    train(backbone, head)

-------

Advanced strategies
===================

Every finetune strategy can also be customized.


freeze_unfreeze
---------------
In this strategy, the backbone is frozen for x epochs then unfrozen.

Here we unfreeze the backbone at epoch 11.

.. code-block:: python

    from flash.core.finetuning import FreezeUnfreeze

    trainer = flash.Trainer(max_epochs=10)
    trainer.finetune(model, data, strategy=FreezeUnfreeze(unfreeze_epoch=11))

unfreeze_milestones
-------------------
This strategy allows you to unfreeze part of the backbone at predetermined intervals

Here's an example where:
- backbone starts frozen
- at epoch 3 the last 2 layers unfreeze
- at epoch 8 the full backbone unfreezes

|

.. code-block:: python

    from flash.core.finetuning import UnfreezeMilestones

    # finetune for 10 epochs.
    trainer = flash.Trainer(max_epochs=10)
    trainer.finetune(model, data, strategy=UnfreezeMilestones(unfreeze_milestones=(3, 8), num_layers=2))

Under the hood, the pseudocode looks like:

.. code-block:: python

    backbone = Resnet50()
    head = nn.Linear(...)

    # freeze backbone
    backbone.freeze()
    head.unfreeze()

    train(backbone, head, epochs=3)

    # unfreeze last 2 layers at epoch 3
    backbone.unfreeze_last_layers(2)

    train(backbone, head, epochs=8)

    # unfreeze the full backbone
    backbone.unfreeze()

--------

Custom Strategy
===============
For even more customization, create your own finetuning callback. Learn more about callbacks `here <https://pytorch-lightning.readthedocs.io/en/stable/callbacks.html>`_.

.. code-block:: python

    from flash.core.finetuning import FlashBaseFinetuning

    # Create a finetuning callback
    class FeatureExtractorFreezeUnfreeze(FlashBaseFinetuning):

        def __init__(self, unfreeze_at_epoch: int = 5, train_bn: bool = true)
            # this will set self.attr_names as ["feature_extractor"]
            super().__init__("feature_extractor", train_bn)
            self._unfreeze_at_epoch = unfreeze_at_epoch

        def finetune_function(self, pl_module, current_epoch, optimizer, opt_idx):
            # unfreeze any module you want by overriding this function

            # When ``current_epoch`` is 5, feature_extractor will start to be trained.
            if current_epoch == self._unfreeze_at_epoch:
                self.unfreeze_and_add_param_group(
                    module=pl_module.feature_extractor,
                    optimizer=optimizer,
                    train_bn=True,
                )

    # Init the trainer
    trainer = flash.Trainer(max_epochs=10)

    # pass the callback to trainer.finetune
    trainer.finetune(model, data, strategy=FeatureExtractorFreezeUnfreeze(unfreeze_epoch=5))
