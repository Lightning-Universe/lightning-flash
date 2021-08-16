
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
