**********
Finetuning
**********

Finetuning (or transfer-learning) is the process of tweaking a model trained on a large dataset, to your particular (likely much smaller) dataset. All Flash tasks have a pre-trained backbone that was trained on large datasets such as ImageNet, and that allows to decrease training time significantly.

Finetuning process can be splitted into 4 steps:

- 1. Train a source neural network model on a source dataset. For computer vision, it is traditionally  the [ImageNet dataset](http://www.image-net.org/search?q=cat). As training is costly, library such as [Torchvion](https://pytorch.org/docs/stable/torchvision/index.html) library supports popular pre-trainer model architectures.

- 2. Create a new neural network called the target model. Its architecture replicates the source model and parameters, expect the latest layer which is removed. This model without its latest layer is traditionally called a backbone

- 3. Add new layers after the backbone where the latest output size is the number of target dataset categories. Those new layers, traditionally called head will be randomly initialized while backbone will conserve its pre-trained weights from ImageNet.

- 4. Train the target model on a smaller target dataset. However, as new layers are randomly initialized, the first gradients will be random when training starts and will destabilize the backbone pre-trained parameters. Therefore, it is good pratice to freeze the backbone, which means the parameters of the backbone won't be trainable for some epochs. After some epochs, the backbone are being unfreezed, meaning the weights will be trainable.


.. tip:: If you have a huge dataset and prefer to train from scratch, see the training guide.

You can finetune any Flash tasks on your own data in just a 3 simple steps:

1. Load your data and organize it using `Flash DataModules`. Note that different tasks have different data modules (The :class:`~flash.vision.ImageClassificationData` for image classification, :class:`~flash.text.TextClassificationData` for text classification, etc.).

2. Pick a model to run from a variety of Flash tasks: :class:`~flash.vision.ImageClassification`, :class:`~flash.text.TextClassifier`, :class:`~flash.tabular.TabularClassifier`, all optimized with the latest best practices.

3. Finetune your model using  :func:`~flash.Trainer.finetune` method. You will need to choose a finetune strategy.


Finetune options
================

Flash supports 4 builts-in Finetuning options:

* `no_freeze`: Don't freeze anything.
* `freeze`: Freeze the backbone parameters when training starts.
* `freeze_unfreeze`: Freeze the backbone parameters when training starts and unfreeze the backbone when reaching `unfreeze_epoch`.
* `unfreeze_milestones`: Freeze the backbone parameters when training starts and unfreeze the end backbone when reaching first milestones and begining when reaching second one.

For more options, you can pass in an instance of :class:`~python_lightning.callbacks.finetuning.BaseFinetuning` to the `strategy` parameter.
Furthermore,


Once training is completed, you can use the model for inference to make predictions using the `predict` method.


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
    trainer.finetune(model, data, strategy="no_freeze")



Flash provides a very simple interface for finetuning through `trainer.finetune` with its `strategy` parameters.

Flash finetune `strategy` argument can either a string or an instance of :class:`~python_lightning.callbacks.finetuning.BaseFinetuning`.
Furthermore, Flash supports 4 builts-in Finetuning Callback accessible via those strings:

* `no_freeze`: Don't freeze anything.
* `freeze`: The parameters of the backbone won't be trainable after training starts.
* `freeze_unfreeze`: The parameters of the backbone won't be trainable when training start and then those parameters will become trainable when training epoch reaches `unfreeze_epoch`.
* `unfreeze_milestones`: The parameters of the backbone won't be trainable when training start. However, the latest layers of the backbone will become trainable when training epoch reaches the first milestone and the remaining layers when reaching the second one.

==========================
Custom callback finetuning
==========================

You can pass in the built in callbacks for more customization:

.. code-block:: python

    # finetune for 10 epochs
    trainer = flash.Trainer()
    trainer.finetune(model, data, strategy="freeze_unfreeze")

    # or import FreezeUnfreeze
    from flash.core.finetuning import FreezeUnfreeze

    # finetune for 10 epochs. Backbone will be frozen for 5 epochs.
    trainer = flash.Trainer()
    trainer.finetune(model, data, strategy=FreezeUnfreeze(unfreeze_epoch=5))



Custom callback finetuning
==========================

For even more customization, create your own finetuning callback.

.. code-block:: python

    from flash.core.finetuning import FlashBaseFinetuning

    # Create a finetuning callback
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

    # Init the trainer
    trainer = flash.Trainer(max_epochs=10)

    # pass the callback to trainer.finetune
    trainer.finetune(model, data, strategy=FeatureExtractorFreezeUnfreeze(unfreeze_epoch=5))
