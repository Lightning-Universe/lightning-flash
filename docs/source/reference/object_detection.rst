
.. _object_detection:

################
Object Detection
################

********
The Task
********

Object detection is the task of identifying objects in images and their associated classes and bounding boxes.

------

*******
Example
*******

Let's look at object detection with the COCO 128 data set, which contains `91 object classes <https://cocodataset.org/#explore>`_.
This is a subset of `COCO train2017 <https://cocodataset.org/>`_ with only 128 images.
The data set is organized following the COCO format.
Here's an outline:

.. code-block::

    coco128
    ├── annotations
    │   └── instances_train2017.json
    ├── images
    │   └── train2017
    │       ├── 000000000009.jpg
    │       ├── 000000000025.jpg
    │       ...
    └── labels
        └── train2017
            ├── 000000000009.txt
            ├── 000000000025.txt
            ...

Once we've downloaded the data using :func:`~flash.core.data.download_data`, we can create the :class:`~flash.image.detection.data.ObjectDetectionData`.
We select a pre-trained RetinaNet to use for our :class:`~flash.image.detection.model.ObjectDetector` and fine-tune on the COCO 128 data.
We then use the trained :class:`~flash.image.detection.model.ObjectDetector` for inference.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/object_detection.py
    :language: python
    :lines: 14-
