
.. _image_embedder:

##############
Image Embedder
##############

********
The task
********
Image embedding encodes an image into a vector of image features which can be used for anything like clustering, similarity
search or classification.

------

*********
Inference
*********

The :class:`~flash.vision.ImageEmbedder` is already pre-trained on `ImageNet <http://www.image-net.org/>`_, a dataset of over 14 million images.

Use the :class:`~flash.vision.ImageEmbedder` pretrained model for inference on any image tensor or image path using :func:`~flash.vision.ImageEmbedder.predict`:

.. code-block:: python

	from flash.vision import ImageEmbedder

	# Load finetuned task
	embedder = ImageEmbedder(backbone="resnet18")

	# 2. Perform inference on an image file
	embeddings = embedder.predict("path/to/image.png")
	print(embeddings)

Or on a random image tensor

.. code-block:: python

    # 2. Perform inference on a random image tensor
    import torch
    images = torch.rand(32, 3, 224, 224)
    embeddings = embedder.predict(images)
    print(embeddings)

For more advanced inference options, see :ref:`predictions`.

------

**********
Finetuning
**********
To tailor this image embedder to your dataset, finetune first.

.. code-block:: python

    import flash
    from flash import download_data
    from flash.vision import ImageClassificationData, ImageEmbedder

    # 1. Download the data
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "data/")

    # 2. Load the data
    datamodule = ImageClassificationData.from_folders(
        train_folder="data/hymenoptera_data/train/",
        valid_folder="data/hymenoptera_data/val/",
        test_folder="data/hymenoptera_data/test/",
    )

    # 3. Build the model
    embedder = ImageEmbedder(backbone="resnet18")

    # 4. Create the trainer. Run once on data
    trainer = flash.Trainer(max_epochs=1)

    # 5. Train the model
    trainer.finetune(embedder, datamodule=datamodule, strategy="freeze_unfreeze")

    # 6. Test the model
    trainer.test()

    # 7. Save it!
    trainer.save_checkpoint("image_embedder_model.pt")

------

*********************
Changing the backbone
*********************
By default, we use the encoder from `SwAV <https://arxiv.org/pdf/2006.09882.pdf>`_ pretrained on Imagenet via contrastive learning. You can change the model run by the task by passing in a different backbone.

.. note::

   When changing the backbone, make sure you pass in the same backbone to the Task!

.. code-block:: python

    # 1. organize the data
    data = ImageClassificationData.from_folders(
        train_folder="data/hymenoptera_data/train/",
        valid_folder="data/hymenoptera_data/val/"
    )

    # 2. build the task
    embedder = ImageEmbedder(backbone="resnet34")

Backbones available

.. list-table:: Backbones
   :widths: 50 20 20
   :header-rows: 1

   * - backbone
     - dataset
     - training method
   * - resnet18
     - Imagenet
     - supervised
   * - resnet34
     - Imagenet
     - supervised
   * - resnet50
     - Imagenet
     - supervised
   * - resnet101
     - Imagenet
     - supervised
   * - resnet152
     - Imagenet
     - supervised
   * - swav-imagenet
     - Imagenet
     - self-supervised (clustering)

------

*************
API reference
*************

.. _image_embedder_class:

ImageEmbedder
---------------

.. autoclass:: flash.vision.ImageEmbedder
    :members:
    :exclude-members: forward

.. _image_embedder_data:
