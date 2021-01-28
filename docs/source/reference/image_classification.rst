
.. _image_classification:


Image Classification
====================

The task
--------
The task of identifying what an image represents is called `image classification <https://arxiv.org/abs/1912.12162>`_. Typically, Image Classification would be used to analyze images that contain a single object. The task will identify the ‘class’ the image falls under, or the probability of the image being part of a ‘class’.  A class is essentially a label, such as ‘car’, ‘house’, ‘cat’ etc. For example, we can train the image classifier task on images of ants and it will learn to predict the probability that an image contains an ant.


Inference
---------

The :class:`~flash.vision.ImageClassificatier` is already pre-trained on [ImageNet](http://www.image-net.org/), a dataset of over 14 million images.


Use the :class:`~flash.text.ImageClassificatier` pretrained model for inference on any string sequence using :func:`~flash.text.TextClassifier.predict`:

.. code-block:: python

	# import our libraries
	from flash.text import TextClassifier


	# Load finetuned task
	model = ImageClassifier.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/image_classification_model.pt")

	# 2. Perform inference from list of sequences
	predictions = model.predict([
	    "Turgid dialogue, feeble characterization - Harvey Keitel a judge?.",
	    "The worst movie in the history of cinema.",
	    "I come from Bulgaria where it 's almost impossible to have a tornado."
	    "Very, very afraid"
	    "This guy has done a great job with this movie!",
	])
	print(predictions)

Or on a given dataset:

.. code-block:: python

	# import our libraries

	from flash.core.data import download_data
	from flash.text import TextClassifier


	# 1. Download dataset, save it under 'data' dir
	download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", 'data/')

	# 2. Load finetuned task
	model = ImageClassifier.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/image_classification_model.pt")

	# 3. Perform inference from a csv file
	predictions = model.predict("data/imdb/test.csv")
	print(predictions)

For more advanced inference options, see :ref:`predictions`.


Finetuning
----------

Lets say you wanted to develope a model that could classify between **ants** and **bees**, using the hymenoptera dataset.
Once we doanload the data using :func:`~flash.data.download_data`, all we need is the train data and validation data folders to create the :class:`~flash.vision.ImageClassificationData`.

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


Now all we need is three lines of code to build and train our task!

.. code-block:: python

	import flash
	from flash.core.data import download_data
	from flash.vision import ImageClassificationData, ImageClassifier


    # 1. Download the data
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", 'data/')

    # 2. Load the data
    datamodule = ImageClassificationData.from_folders(
    	backbone="resnet18",
        train_folder="data/hymenoptera_data/train/",
        valid_folder="data/hymenoptera_data/val/",
        test_folder="data/hymenoptera_data/test/",
    )

    # 3. Build the model
    model = ImageClassifier(num_classes=datamodule.num_classes)

    # 4. Create the trainer. Run once on data
    trainer = flash.Trainer(max_epochs=1)

    # 5. Train the model
    trainer.finetune(model, datamodule=datamodule, unfreeze_milestones=(0, 1))

    # 6. Test the model
    trainer.test()

    # 7. Save it!
    trainer.save_checkpoint("image_classification_model.pt")



Changing the backbone
=====================
By default, we use a `ResNet-18 <https://arxiv.org/abs/1512.03385>`_ for image classification. You can change the model run by the task by passing in a different backbone.

.. note:: When changing the backbone, make sure you pass in the same backbone to the Task and the Data object!

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

* resnet34
* resnet50
* resnet101
* resnet152

------


API reference
=============

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

.. automethod:: flash.vision.ImageClassificationData.from_filepaths

.. automethod:: flash.vision.ImageClassificationData.from_folders




