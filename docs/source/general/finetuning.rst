**********
Finetuning
**********

Finetuning (or transfer-learning) is the process of tweaking a model trained on a large dataset, to your particular (likely much smaller) dataset. All Flash tasks have a pre-trained backbone that was trained on large datasets such as ImageNet, and that allows to decrease training time significantly.

Finetuning process can be splitted into 4 steps:

- 1. Train a source neural network model on a source dataset. For computer vision, it is traditionally  the [ImageNet dataset](http://www.image-net.org/search?q=cat). As training is costly, library such as [Torchvion](https://pytorch.org/docs/stable/torchvision/index.html) library supports popular pre-trainer model architectures.

- 2. Create a new neural network called the target model. Its architecture replicates the source model and parameters, expect the latest layer which is removed. This model without its latest layer is traditionally called a backbone

- 3. Add new layers after the backbone where the latest output size is the number of target dataset categories. Those new layers, traditionally called head will be randomly initialized while backbone will conserve its pre-trained weights from ImageNet.

- 4. Train the target model on a smaller target dataset. However, as new layers are randomly initialized, the first gradients will be random when training starts and will destabilize the backbone pre-trained parameters. Therefore, it is good pratice to freeze the backbone, which means the parameters of the backbone won't be trainable for some epochs. After some epochs, the backbone are being unfreezed, meaning the weights will be trainable.


.. tip:: If you have a huge dataset and prefer to train from scratch.

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
    task = ImageClassifier(num_classes=2)

    # 3. train!
    trainer = flash.Trainer()
    trainer.finetune(model, data, strategy="no_freeze")


Finetune options
================

Flash provides a very simple interface for finetuning through `trainer.finetune` with its `strategy` parameters.

Flash finetune `strategy` argument can either a string or an instance of :class:`~python_lightning.callbacks.finetuning.BaseFinetuning`.
Furthermore, Flash supports 4 builts-in Finetuning Callback accessible via those strings:

* `no_freeze`: Don't freeze anything.
* `freeze`: The parameters of the backbone won't be trainable after training starts.
* `freeze_unfreeze`: The parameters of the backbone won't be trainable when training start and then those parameters will become trainable when training epoch reaches `unfreeze_epoch`.
* `unfreeze_milestones`: The parameters of the backbone won't be trainable when training start. However, the latest layers of the backbone will become trainable when training epoch reaches the first milestone and the remaining layers when reaching the second one.

Use the built-in Finetuning Strategy Callbacks.

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

        def __init__(self, unfreeze_at_epoch: int = 5, train_bn: bool = True)
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

    trainer = flash.Trainer(max_epochs=10)
    trainer.finetune(model, data, strategy=FeatureExtractorFreezeUnfreeze(unfreeze_epoch=5))
