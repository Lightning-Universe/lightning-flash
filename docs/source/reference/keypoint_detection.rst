.. customcarditem::
   :header: Keypoint Detection
   :card_description: Learn to detect keypoints in images with Flash and build a network to detect facial keypoints with the BIWI data set.
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/keypoint_detection.svg
   :tags: Image,Keypoint,Detection

.. _keypoint_detection:

##################
Keypoint Detection
##################

********
The Task
********

Keypoint detection is the task of identifying keypoints in images and their associated classes.

The :class:`~flash.image.keypoint_detection.model.KeypointDetector` and :class:`~flash.image.keypoint_detection.data.KeypointDetectionData` classes internally rely on `IceVision <https://airctic.com/>`_.

------

*******
Example
*******

Let's look at keypoint detection with `BIWI Sample Keypoints (center of face) <https://www.kaggle.com/kmader/biwi-kinect-head-pose-database>`_ from `IceData <https://github.com/airctic/icedata>`_.
Once we've downloaded the data, we can create the :class:`~flash.image.keypoint_detection.data.KeypointDetectionData`.
We select a ``keypoint_rcnn`` with a ``resnet18_fpn`` backbone to use for our :class:`~flash.image.keypoint_detection.model.KeypointDetector` and fine-tune on the BIWI data.
We then use the trained :class:`~flash.image.keypoint_detection.model.KeypointDetector` for inference.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/keypoint_detection.py
    :language: python
    :lines: 14-

To learn how to the available backbones / heads for this task, see :ref:`backbones_heads`.

------

**********
Flash Zero
**********

The keypoint detector can be used directly from the command line with zero code using :ref:`flash_zero`.
You can run the above example with:

.. code-block:: bash

    flash keypoint_detection

To view configuration options and options for running the keypoint detector with your own data, use:

.. code-block:: bash

    flash keypoint_detection --help
