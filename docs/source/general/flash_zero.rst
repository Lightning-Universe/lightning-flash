.. _flash_zero:

**********
Flash Zero
**********

Flash Zero is a zero-code machine learning platform.
Here's an image classification example to illustrate with one of the dozens tasks available.


Flash Zero in 3 steps
_____________________

1. Select your task
===================

.. code-block:: bash

    flash {TASK_NAME}

Here is the list of currently supported tasks.

.. code-block:: bash

    audio_classification     Classify audio spectrograms.
    graph_classification     Classify graphs.
    image_classification     Classify images.
    instance_segmentation    Segment object instances in images.
    keypoint_detection       Detect keypoints in images.
    object_detection         Detect objects in images.
    pointcloud_detection     Detect objects in point clouds.
    pointcloud_segmentation  Segment objects in point clouds.
    question_answering       Extractive Question Answering.
    semantic_segmentation    Segment objects in images.
    speech_recognition       Speech recognition.
    style_transfer           Image style transfer.
    summarization            Summarize text.
    tabular_classification   Classify tabular data.
    text_classification      Classify text.
    translation              Translate text.
    video_classification     Classify videos.


2. Pass in your own data
========================

.. code-block:: bash

    flash image_classification from_folders --train_folder data/hymenoptera_data/train


3. Modify the model and training parameters
===========================================

.. code-block:: bash

    flash image_classification --trainer.max_epochs 10 --model.backbone resnet50 from_folders --train_folder data/hymenoptera_data/train

.. note::

    The trainer and model arguments should be placed before the ``source`` subcommand. Here it is ``from_folders``.


Other Examples
______________

Image Object Detection
======================

To train an Object Detector on `COCO 2017 dataset <https://cocodataset.org/>`_, you could use the following command:

.. code-block:: bash

    flash object_detection from_coco --train_folder data/coco128/images/train2017/ --train_ann_file data/coco128/annotations/instances_train2017.json --val_split .3 --batch_size 8 --num_workers 4


Image Object Segmentation
=========================

To train an Image Segmenter on `CARLA driving simulator dataset <http://carla.org/>`_

.. code-block:: bash

    flash semantic_segmentation from_folders --train_folder data/CameraRGB --train_target_folder data/CameraSeg --num_classes 21

Below is an example where the head, the backbone and its pretrained weights are customized.

.. code-block:: bash

    flash semantic_segmentation --model.head fpn --model.backbone efficientnet-b0 --model.pretrained advprop from_folders --train_folder data/CameraRGB --train_target_folder data/CameraSeg --num_classes 21

Video Classification
====================

To train an Video Classifier on the `Kinetics dataset <https://deepmind.com/research/open-source/kinetics>`_, you could use the following command:


.. code-block:: bash

    flash video_classification from_folders --train_folder data/kinetics/train/ --clip_duration 1 --num_workers 0


CLI options
___________

Flash Zero is built on top of the
`lightning CLI <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html>`_, so the trainer and
model arguments can be configured either from the command line or from a config file.
For example, to run the image classifier for 10 epochs with a `resnet50` backbone you can use:

.. code-block:: bash

    flash image_classification --trainer.max_epochs 10 --model.backbone resnet50

To view all of the available options for a task, run:

.. code-block:: bash

    flash image_classification --help

Using Your Own Data
___________________

Flash Zero works with your own data through subcommands. The available subcommands for each task are given at the bottom
of their help pages (e.g. when running :code:`flash image-classification --help`). You can then use the required
subcommand to train on your own data. Let's look at an example using the Hymenoptera data from the
:ref:`image_classification` guide. First, download and unzip your data:

.. code-block:: bash

    curl https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip -o hymenoptera_data
    unzip hymenoptera_data.zip

Now train with Flash Zero:

.. code-block:: bash

    flash image_classification from_folders --train_folder ./hymenoptera_data/train

Getting Help
____________

To find all available tasks, you can run:

.. code-block:: bash

    flash --help

This will output the following:

.. code-block:: bash

    Commands:
    audio_classification     Classify audio spectrograms.
    graph_classification     Classify graphs.
    image_classification     Classify images.
    instance_segmentation    Segment object instances in images.
    keypoint_detection       Detect keypoints in images.
    object_detection         Detect objects in images.
    pointcloud_detection     Detect objects in point clouds.
    pointcloud_segmentation  Segment objects in point clouds.
    question_answering       Extractive Question Answering.
    semantic_segmentation    Segment objects in images.
    speech_recognition       Speech recognition.
    style_transfer           Image style transfer.
    summarization            Summarize text.
    tabular_classification   Classify tabular data.
    text_classification      Classify text.
    translation              Translate text.
    video_classification     Classify videos.


To get more information about a specific task, you can do the following:

.. code-block:: bash

    flash image_classification --help

You can view the help page for each subcommand. For example, to view the options for training an image classifier from
folders, you can run:

.. code-block:: bash

    flash image_classification from_folders --help

Finally, you can generate a `config.yaml` file from the client to ease parameters modification by running:

.. code-block:: bash

    flash image_classification --print_config > config.yaml
