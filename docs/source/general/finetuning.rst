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

Finetuning in Flash
===================

From the :ref:`quick_start` guide.

.. include:: ../common/finetuning_example.rst

------

Finetune strategies
===================

.. testsetup:: strategies

    import flash
    from flash.core.data.utils import download_data
    from flash.image import ImageClassificationData, ImageClassifier

    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "data/")

    datamodule = ImageClassificationData.from_files(
        train_files=["data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg"],
        train_targets=[0],
        batch_size=1,
        num_workers=0,
    )

    model = ImageClassifier(backbone="resnet18", num_classes=2)
    trainer = flash.Trainer(max_epochs=1, checkpoint_callback=False)

Finetuning is very task specific. Each task encodes the best finetuning practices for that task.
However, Flash gives you a few default strategies for finetuning.

Finetuning operates on two things, the model backbone and the head. The backbone
is the neural network that was pre-trained. The head is another neural network that bridges between the backbone
and your particular dataset.

no_freeze
---------
In this strategy, the backbone and the head are unfrozen from the beginning.

.. testcode:: strategies

    trainer.finetune(model, datamodule, strategy="no_freeze")

.. testoutput:: strategies
    :hide:

    ...

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

.. testcode:: strategies

    trainer.finetune(model, datamodule, strategy="freeze")

The pseudocode looks like:

.. code-block:: python

    backbone = Resnet50()
    head = nn.Linear(...)

    # freeze backbone
    backbone.freeze()
    head.unfreeze()

    train(backbone, head)

-------

Advanced strategies
===================

Every finetune strategy can also be customized.


freeze_unfreeze
---------------
The freeze_unfreeze strategy keeps the backbone frozen until a certain epoch (provided through the input) after which the backbone unfrozen.

For example, to unfreeze after epoch 7:

.. testcode:: strategies

    trainer.finetune(model, datamodule, strategy=("freeze_unfreeze", 7))

Under the hood, the pseudocode looks like:

.. code-block:: python

    backbone = Resnet50()
    head = nn.Linear(...)

    # freeze backbone
    backbone.freeze()
    head.unfreeze()

    train(backbone, head, epochs=10)

    # unfreeze after 7 epochs
    backbone.unfreeze()

    train(backbone, head)

unfreeze_milestones
-------------------
This strategy allows you to unfreeze part of the backbone at predetermined intervals

Here's an example where:

* backbone starts frozen
* at epoch 3 the last 2 layers unfreeze
* at epoch 8 the full backbone unfreezes

|

.. testcode:: strategies

    trainer.finetune(model, datamodule, strategy=("unfreeze_milestones", ((3, 8), 2)))

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

.. testcode:: strategies

    from flash.core.finetuning import FlashBaseFinetuning

    # Create a finetuning callback
    class FeatureExtractorFreezeUnfreeze(FlashBaseFinetuning):
        def __init__(self, unfreeze_epoch: int = 5, train_bn: bool = True):
            # this will set self.attr_names as ["backbone"]
            super().__init__("backbone", train_bn)
            self._unfreeze_epoch = unfreeze_epoch

        def finetune_function(self, pl_module, current_epoch, optimizer, opt_idx):
            # unfreeze any module you want by overriding this function

            # When ``current_epoch`` is 5, backbone will start to be trained.
            if current_epoch == self._unfreeze_epoch:
                self.unfreeze_and_add_param_group(
                    pl_module.backbone,
                    optimizer,
                )


    # Pass the callback to trainer.finetune
    trainer.finetune(model, datamodule, strategy=FeatureExtractorFreezeUnfreeze(unfreeze_epoch=5))
