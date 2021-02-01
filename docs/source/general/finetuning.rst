**********
Finetuning
**********

When training a Flash task, calling the :func:`~flash.Trainer.finetune` using a pre-trained backbone will finetune the backbone onto your data and desired task. If you would like to train from scratch, pass `pretrained=False` when creating your task whilst using the :func:`~flash.Trainer.fit` method to start training.

When using :func:`~flash.Trainer.finetune`, you also need to provide a finetune `strategy`.

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

    # 2. build the task, and turn off pre-training
    task = ImageClassifier(num_classes=2, pretrained=False)

    # 3. train!
    trainer = flash.Trainer()
    trainer.finetune(model, data, strategy="no_freeze")


Finetune options
================

Flash finetune `strategy` argument can either a string or an instance of :class:`~python_lightning.callbacks.finetuning.BaseFinetuning`.
Furthermore, Flash supports 4 builts-in Finetuning Callback accessible via those strings:

* `no_freeze: Don't freeze anything.
* `freeze`: Freeze the backbone parameters when training starts.
* `freeze_unfreeze`: Freeze the backbone parameters when training starts and unfreeze the backbone when reaching `unfreeze_epoch`.
* `unfreeze_milestones`: Freeze the backbone parameters when training starts and unfreeze the end backbone when reaching first milestones and begining when reaching second one.

Use the builts-in Finetuning Strategy Callback.

.. code-block:: python

    # finetune for 10 epochs
    trainer = flash.Trainer()
    trainer.finetune(model, data, strategy="freeze_unfreeze")

    # or import FreezeUnfreeze
    from flash.core.finetuning import FreezeUnfreeze

    # finetune for 10 epochs. Backbone will be frozen for 5 epochs.
    trainer = flash.Trainer()
    trainer.finetune(model, data, strategy=FreezeUnfreeze(unfreeze_epoch=5))

Create a custom Finetuning Strategy Callback.

.. code-block:: python

    from flash.core.finetuning import FlashBaseFinetuning

    class FeatureExtractorFreezeUnfreeze(FlashBaseFinetuning):

        def __init__(self, unfreeze_at_epoch: int = 5, train_bn: bool = true)
            # this will set self.attr_names as ["feature_extractor"]
            super().__init__("feature_extractor", train_bn)
            self._unfreeze_at_epoch = unfreeze_at_epoch

        def freeze_before_training(self, pl_module):
            # freeze any module you want by overriding this function

            # Here, we are freezing ``feature_extractor``
            self.freeze_using_attr_names(pl_module, self.attr_names, train_bn=self.train_bn)

        def finetune_function(self, pl_module, current_epoch, optimizer, opt_idx):
            # unfreeze any module you want by overriding this function

            # When ``current_epoch`` is 5, feature_extractor will start to be trained.
            if current_epoch == self._unfreeze_at_epoch:
                self.unfreeze_and_add_param_group(
                    module=pl_module.feature_extractor,
                    optimizer=optimizer,
                    train_bn=True,
                )

    trainer = flash.Trainer(max_epochs=10)
    trainer.finetune(model, data, strategy=FeatureExtractorFreezeUnfreeze(unfreeze_epoch=5))
