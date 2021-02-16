.. _finetuning:

**********
Finetuning
**********

Finetuning (or transfer-learning) is the process of tweaking a model trained on a large dataset, to your particular (likely much smaller) dataset. All Flash tasks have a pre-trained backbone that was already trained on large datasets such as ImageNet. Finetuning on already pretrained models decrease training time significantly.

You can finetune any Flash task on your own data in just a 3 simple steps:

1. Load your data and organize it using Flash Data Modules. Note that different tasks have different data modules (The :class:`~flash.vision.ImageClassificationData` for image classification, :class:`~flash.text.classification.data.TextClassificationData` for text classification, etc.).

2. Pick a model to run from a variety of Flash tasks: :class:`~flash.vision.ImageClassifier`, :class:`~flash.text.classification.model.TextClassifier`, :class:`~flash.tabular.TabularClassifier`, all optimized with the latest best practices.

3. Finetune your model using :func:`~flash.core.trainer.Trainer.finetune` method. You will need to choose a finetune strategy.

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

.. tip:: If you have a large dataset and prefer to train from scratch, see the :ref:`training` guide.


Finetune strategies
===================

The flash tasks contain pre-trained models trained on large datasets such as `ImageNet <http://www.image-net.org/>`_, which contains millions of images. These models are called **backbones**. This will be used as the starting point for finetuning.

The model needs to be adapted or refined for the new data available for the task. Usually, the last layers of the backbone need to be modified, to match the backbone output to the number of target classes of the new data. These layers are commonly referred to as the **head**.
For example, our backbone might be trained to classify 10 types of animals, but maybe our new dataset only contains images of bees and ants, so we would have to modify our final layer to fit just 2 classes.
The head is randomly initialized whereas the backbone conserves its pre-trained weights.

The :func:`~flash.core.trainer.Trainer.finetune` method trains the new modified model using the new dataset. As the head (new layers) is untrained, the first results (gradients) will be random when training starts and could decrease the backbone performance (by changing its pre-trained parameters). Therefore, it is a good practice to "freeze" the backbone, meaning the parameters of the backbone won't be updated until they are "unfrozen" a few epochs later.

You can choose a finetuning strategy using :func:`~flash.core.trainer.Trainer.finetune` `strategy` parameter. Flash finetune `strategy` argument can either a string or an instance of :class:`~flash.core.finetuning.FlashBaseFinetuning`.

Flash supports 2 builts-in Finetuning strategies, that can be passed as strings:

* `no_freeze`: Don't freeze anything, the backbone parameters can be modified during finetuning.
* `freeze`: The parameters of the backbone won't be modified during finetuning.

.. code-block:: python

    # using the freeze strategy
    trainer.finetune(model, data, strategy="freeze")

    # using the no_freeze strategy
    trainer.finetune(model, data, strategy="no_freeze")


For more options, you can pass in an instance of :class:`~python_lightning.callbacks.finetuning.BaseFinetuning` to the `strategy` parameter.


==========================
Custom callback finetuning
==========================

For more advanced finetuning, you can use flash built-in finetuning callbacks.

* :class:`~flash.core.finetuning.FreezeUnfreeze`: The backbone parameters will be frozen for a given number of epochs (by default the `unfreeze_epoch` is set to 10).


.. code-block:: python

    # import FreezeUnfreeze
    from flash.core.finetuning import FreezeUnfreeze

    # finetune for 10 epochs. Backbone will be frozen for 5 epochs.
    trainer = flash.Trainer(max_epochs=10)
    trainer.finetune(model, data, strategy=FreezeUnfreeze(unfreeze_epoch=5))

* :class:`~flash.core.finetuning.UnfreezeMilestones`: This strategy define 2 milestones, one milestone (epoch number) to unfreeze the last layers of the backbone, and a second milestone to unfreeze the remaining layers. For example, by default the first milestone is 5 and the second is 10. So for the first 4 epochs, the backbone parameters will be frozen. In epochs 5-9, only the last layers (5 by deafult) can be trained. After the 10thg epoch, all parameters in all layers can be trained.


.. code-block:: python

    # import UnfreezeMilestones
    from flash.core.finetuning import UnfreezeMilestones

    # finetune for 10 epochs. Backbone will be frozen for 3 epochs. The last 2 layers will be unfrozen for the first 4 epochs,
    # and then the rest will be unfrozen on the 8th epoch
    trainer = flash.Trainer(max_epochs=10)
    trainer.finetune(model, data, strategy=UnfreezeMilestones(unfreeze_milestones=(5,8), num_layers=2))


Custom callback finetuning
==========================

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
