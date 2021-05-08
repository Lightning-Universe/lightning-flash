
.. _semantinc_segmentation:

######################
Semantinc Segmentation
######################

********
The task
********
Semantic segmentation, or image segmentation, is the task of clustering parts of an image together which belong to the same object class. It is a form of pixel-level prediction because each pixel in an image is classified according to a category

See more: https://paperswithcode.com/task/semantic-segmentation

.. raw:: html

   <p>
     <a href="https://i2.wp.com/syncedreview.com/wp-content/uploads/2019/12/image-9-1.png" >
       <img src="https://i2.wp.com/syncedreview.com/wp-content/uploads/2019/12/image-9-1.png"/>
     </a>
   </p>

------

*********
Inference
*********

The :class:`~flash.vision.SemanticSegmentation` is already pre-trained on a generated dataset from `CARLA <http://carla.org/>`_ simulator.


Use the :class:`~flash.vision.SemanticSegmentation` pretrained model for inference on any string sequence using :func:`~flash.vision.SemanticSegmentation.predict`:

.. code-block:: python

    # import our libraries
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

    import flash
    from flash.data.utils import download_data
    from flash.vision import SemanticSegmentation, SemanticSegmentationData
    from flash.vision.segmentation.serialization import SegmentationLabels

    # 1. Download the data
    download_data(
        "https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180513A.zip",
        "data/"
    )

    # 2.1 Load the data
    datamodule = SemanticSegmentationData.from_folders(
        train_folder="data/CameraRGB",
        train_target_folder="data/CameraSeg",
        batch_size=4,
        val_split=0.3,
        image_size=(200, 200),  # (600, 800)
    )

    # 2.2 Visualise the samples
    labels_map = SegmentationLabels.create_random_labels_map(num_classes=21)
    datamodule.set_labels_map(labels_map)
    datamodule.show_train_batch(["load_sample", "post_tensor_transform"])

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

.. automethod:: flash.vision.SemanticSegmentationData.from_folders

.. autoclass:: flash.vision.SemanticSegmentationPreprocess
