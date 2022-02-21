.. customcarditem::
   :header: Object Detection
   :card_description: Learn to detect objects in images with Flash and build an example detector with the COCO data set.
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/object_detection.svg
   :tags: Image,Detection

.. _object_detection:

################
Object Detection
################

********
The Task
********

Object detection is the task of identifying objects in images and their associated classes and bounding boxes.

The :class:`~flash.image.detection.model.ObjectDetector` and :class:`~flash.image.detection.data.ObjectDetectionData` classes internally rely on `IceVision <https://airctic.com/>`_.

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
We select a pre-trained EfficientDet to use for our :class:`~flash.image.detection.model.ObjectDetector` and fine-tune on the COCO 128 data.
We then use the trained :class:`~flash.image.detection.model.ObjectDetector` for inference.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/object_detection.py
    :language: python
    :lines: 14-

To learn how to view the available backbones / heads for this task, see :ref:`backbones_heads`.

------

**********
Flash Zero
**********

The object detector can be used directly from the command line with zero code using :ref:`flash_zero`.
You can run the above example with:

.. code-block:: bash

    flash object_detection

To view configuration options and options for running the object detector with your own data, use:

.. code-block:: bash

    flash object_detection --help

------

**********************
Custom Transformations
**********************

Flash automatically applies some default image / mask transformations and augmentations, but you may wish to customize these for your own use case.
The base :class:`~flash.core.data.io.input_transform.InputTransform` defines 7 hooks for different stages in the data loading pipeline.
For object-detection tasks, you can leverage the transformations from `Albumentations <https://github.com/albumentations-team/albumentations>`__ with the :class:`~flash.core.integrations.icevision.transforms.IceVisionTransformAdapter`,
creating a subclass of :class:`~flash.core.data.io.input_transform.InputTransform`

.. code-block:: python

    from dataclasses import dataclass
    import albumentations as alb
    from icevision.tfms import A

    from flash import InputTransform
    from flash.core.integrations.icevision.transforms import IceVisionTransformAdapter
    from flash.image import ObjectDetectionData


    @dataclass
    class BrightnessContrastTransform(InputTransform):
        image_size: int = 128

        def per_sample_transform(self):
            return IceVisionTransformAdapter(
                [*A.aug_tfms(size=self.image_size), A.Normalize(), alb.RandomBrightnessContrast()]
            )


    datamodule = ObjectDetectionData.from_coco(
        train_folder="data/coco128/images/train2017/",
        train_ann_file="data/coco128/annotations/instances_train2017.json",
        val_split=0.1,
        train_transform=BrightnessContrastTransform,
        batch_size=4,
    )
