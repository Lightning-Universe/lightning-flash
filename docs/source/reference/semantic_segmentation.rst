
.. _semantinc_segmentation:

######################
Semantinc Segmentation
######################

********
The task
********
Semantic segmentation, or image segmentation, is the task of clustering parts of an image together which belong to the same object class. It is a form of pixel-level prediction because each pixel in an image is classified according to a category

------

*********
Inference
*********

The :class:`~flash.vision.SemanticSegmentation` is already pre-trained on a generated dataset from `CARLA <http://carla.org/>`_ simulator.


Use the :class:`~flash.vision.SemanticSegmentation` pretrained model for inference on any string sequence using :func:`~flash.vision.SemanticSegmentation.predict`:

.. code-block:: python

    # import our libraries
    from flash import Trainer
    from flash.data.utils import download_data
    from flash.vision import SemanticSegmentation
    from flash.vision.segmentation.serialization import SegmentationLabels

    # 1. Download the data
    download_data(
        "https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180513A.zip",
        "data/"
    )

    # 2. Load the model from a checkpoint
    model = SemanticSegmentation.load_from_checkpoint(
        "https://flash-weights.s3.amazonaws.com/semantic_segmentation_model.pt"
    )
    model.serializer = SegmentationLabels(visualize=True)

    # 3. Predict what's on a few images and visualize!
    predictions = model.predict([
        'data/CameraRGB/F61-1.png',
        'data/CameraRGB/F62-1.png',
        'data/CameraRGB/F63-1.png',
    ])

For more advanced inference options, see :ref:`predictions`.

------

**********
Finetuning
**********

you now want to customise your model with new data using the same dataset.
Once we download the data using :func:`~flash.data.download_data`, all we need is the train data and validation data folders to create the :class:`~flash.vision.SemanticSegmentationData`.

.. note:: the dataset is structured in a way that each sample (an image and its corresponding labels) is stored in separated directories but keeping the same filename.

.. code-block::

    data
    ├── CameraRGB
    │   ├── F61-1.png
    │   ├── F61-2.png
    │       ...
    └── CameraSeg
        ├── F61-1.png
        ├── F61-2.png
            ...


Now all we need is three lines of code to build to train our task!

.. code-block:: python

    import os
    import flash
    from flash.data.utils import download_data
    from flash.vision import SemanticSegmentation, SemanticSegmentationData

    # 1. Download the data
    download_data(
        "https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180513A.zip",
        "data/"
    )

    # 2.1 Load the data


    def load_data(data_root: str = 'data/'):
        images = []
        labels = []
        rgb_path = os.path.join(data_root, "CameraRGB")
        seg_path = os.path.join(data_root, "CameraSeg")
        for fname in os.listdir(rgb_path):
            images.append(os.path.join(rgb_path, fname))
            labels.append(os.path.join(seg_path, fname))
        return images, labels


    images_filepaths, labels_filepaths = load_data()

    # create the data module
    datamodule = SemanticSegmentationData.from_filepaths(
        train_filepaths=images_filepaths,
        train_labels=labels_filepaths,
    )

    # 3. Build the model
    model = SemanticSegmentation(backbone="torchvision/fcn_resnet50", num_classes=21)

    # 4. Create the trainer.
    trainer = flash.Trainer(max_epochs=1)

    # 5. Train the model
    trainer.finetune(model, datamodule=datamodule, strategy='freeze')

    # 7. Save it!
    trainer.save_checkpoint("semantic_segmentation_model.pt")

------

*************
API reference
*************

.. _segmentation:

SemanticSegmentation
--------------------

.. autoclass:: flash.vision.SemanticSegmentation
    :members:
    :exclude-members: forward

.. _segmentation_data:

SemanticSegmentationData
------------------------

.. autoclass:: flash.vision.SemanticSegmentationData

.. automethod:: flash.vision.SemanticSegmentationData.from_filepaths

.. autoclass:: flash.vision.SemanticSegmentationPreprocess
