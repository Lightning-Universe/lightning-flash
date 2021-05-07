
.. _image_classification:

####################
Image Classification
####################

********
The task
********
The task of identifying what is in an image is called image classification. Typically, Image Classification is used to identify images containing a single object. The task predicts which ‘class’ the image most likely belongs to with a degree of certainty.  A class is a label that describes what is in an image, such as ‘car’, ‘house’, ‘cat’ etc. For example, we can train the image classifier task on images of ants and it will learn to predict the probability that an image contains an ant.

------

*********
Inference
*********

The :class:`~flash.vision.ImageClassifier` is already pre-trained on `ImageNet <http://www.image-net.org/>`_, a dataset of over 14 million images.


Use the :class:`~flash.vision.ImageClassifier` pretrained model for inference on any string sequence using :func:`~flash.vision.ImageClassifier.predict`:

.. code-block:: python

    # import our libraries
    from flash import Trainer
    from flash.data.utils import download_data
    from flash.vision import ImageClassificationData, ImageClassifier

    # 1. Download the data
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "data/")

    # 2. Load the model from a checkpoint
    model = ImageClassifier.load_from_checkpoint(
        "https://flash-weights.s3.amazonaws.com/image_classification_model.pt"
    )

    # 3a. Predict what's on a few images! ants or bees?
    predictions = model.predict([
        "data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg",
        "data/hymenoptera_data/val/bees/590318879_68cf112861.jpg",
        "data/hymenoptera_data/val/ants/540543309_ddbb193ee5.jpg",
    ])
    print(predictions)

    # 3b. Or generate predictions with a whole folder!
    datamodule = ImageClassificationData.from_folders(predict_folder="data/hymenoptera_data/predict/")
    predictions = Trainer().predict(model, datamodule=datamodule)
    print(predictions)

For more advanced inference options, see :ref:`predictions`.

------

**********
Finetuning
**********

Lets say you wanted to develope a model that could determine whether an image contains **ants** or **bees**, using the hymenoptera dataset.
Once we download the data using :func:`~flash.data.download_data`, all we need is the train data and validation data folders to create the :class:`~flash.vision.ImageClassificationData`.

.. note:: The dataset contains ``train`` and ``validation`` folders, and then each folder contains a **bees** folder, with pictures of bees, and an **ants** folder with images of, you guessed it, ants.

.. code-block::

    hymenoptera_data
    ├── train
    │   ├── ants
    │   │   ├── 0013035.jpg
    │   │   ├── 1030023514_aad5c608f9.jpg
    │   │   ...
    │   └── bees
    │       ├── 1092977343_cb42b38d62.jpg
    │       ├── 1093831624_fb5fbe2308.jpg
    │       ...
    └── val
        ├── ants
        │   ├── 10308379_1b6c72e180.jpg
        │   ├── 1053149811_f62a3410d3.jpg
        │   ...
        └── bees
            ├── 1032546534_06907fe3b3.jpg
            ├── 10870992_eebeeb3a12.jpg
            ...


Now all we need is three lines of code to build to train our task!

.. code-block:: python

    import flash
    from flash.data.utils import download_data
    from flash.vision import ImageClassificationData, ImageClassifier

    # 1. Download the data
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "data/")

    # 2. Load the data
    datamodule = ImageClassificationData.from_folders(
        train_folder="data/hymenoptera_data/train/",
        valid_folder="data/hymenoptera_data/val/",
        test_folder="data/hymenoptera_data/test/",
    )

    # 3. Build the model
    model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)

    # 4. Create the trainer. Run once on data
    trainer = flash.Trainer(max_epochs=1)

    # 5. Train the model
    trainer.finetune(model, datamodule=datamodule, strategy="freeze_unfreeze")

    # 6. Test the model
    trainer.test()

    # 7. Save it!
    trainer.save_checkpoint("image_classification_model.pt")

------

*********************
Changing the backbone
*********************
By default, we use a `ResNet-18 <https://arxiv.org/abs/1512.03385>`_ for image classification. You can change the model run by the task by passing in a different backbone.

.. note::

    When changing the backbone, make sure you pass in the same backbone to the Task and the Data object!

.. code-block:: python

    # 1. organize the data
    data = ImageClassificationData.from_folders(
        backbone="resnet34",
        train_folder="data/hymenoptera_data/train/",
        valid_folder="data/hymenoptera_data/val/"
    )

    # 2. build the task
    task = ImageClassifier(num_classes=2, backbone="resnet34")

Available backbones:

* resnet18 (default)
* resnet34
* resnet50
* resnet101
* resnet152
* resnext50_32x4d
* resnext101_32x8d
* mobilenet_v2
* vgg11
* vgg13
* vgg16
* vgg19
* densenet121
* densenet169
* densenet161
* swav-imagenet
* `TIMM <https://rwightman.github.io/pytorch-image-models/>`_ (130+ PyTorch Image Models)

------

*************
API reference
*************

.. _image_classifier:

ImageClassifier
---------------

.. autoclass:: flash.vision.ImageClassifier
    :members:
    :exclude-members: forward

.. _image_classification_data:

ImageClassificationData
-----------------------

.. autoclass:: flash.vision.ImageClassificationData

.. autoclass:: flash.vision.ImageClassificationPreprocess
