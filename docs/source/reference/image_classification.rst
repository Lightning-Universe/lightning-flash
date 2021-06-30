
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

The :class:`~flash.image.ImageClassifier` is already pre-trained on `ImageNet <http://www.image-net.org/>`_, a dataset of over 14 million images.


Use the :class:`~flash.image.ImageClassifier` pretrained model for inference on any string sequence using :func:`~flash.image.ImageClassifier.predict`:

.. literalinclude:: ../../../flash_examples/predict/image_classification.py
    :language: python
    :lines: 14-

For more advanced inference options, see :ref:`predictions`.

------

**********
Finetuning
**********

Lets say you wanted to develope a model that could determine whether an image contains **ants** or **bees**, using the hymenoptera dataset.
Once we download the data using :func:`~flash.core.data.download_data`, all we need is the train data and validation data folders to create the :class:`~flash.image.ImageClassificationData`.

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


Now all we need is to train our task!

.. literalinclude:: ../../../flash_examples/finetuning/image_classification.py
    :language: python
    :lines: 14-

------

*********************
Changing the backbone
*********************
By default, we use a `ResNet-18 <https://arxiv.org/abs/1512.03385>`_ for image classification. You can change the model run by the task by passing in a different backbone.

.. testsetup::

    from flash.core.data.utils import download_data
    from flash.image import ImageClassificationData, ImageClassifier

    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "data/")

.. testcode::

    # 1. organize the data
    data = ImageClassificationData.from_folders(
        train_folder="data/hymenoptera_data/train/",
        val_folder="data/hymenoptera_data/val/",
    )

    # 2. build the task
    task = ImageClassifier(num_classes=2, backbone="resnet34")

.. include:: ../common/image_backbones.rst
