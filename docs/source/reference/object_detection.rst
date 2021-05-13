
.. _object_detection:

################
Object Detection
################

********
The task
********

The object detection task identifies instances of objects of a certain class within an image.

------

*********
Inference
*********

The :class:`~flash.vision.ObjectDetector` is already pre-trained on `COCO train2017 <https://cocodataset.org/>`_, a dataset with `91 classes <https://cocodataset.org/#explore>`_ (123,287 images, 886,284 instances).

.. code-block::

    annotation{
        "id": int,
        "image_id": int,
        "category_id": int,
        "segmentation": RLE or [polygon],
        "area": float,
        "bbox": [x,y,width,height],
        "iscrowd": 0 or 1,
    }

    categories[{
        "id": int,
        "name": str,
        "supercategory": str,
    }]

Use the :class:`~flash.vision.ObjectDetector` pretrained model for inference on any image tensor or image path using :func:`~flash.vision.ObjectDetector.predict`:

.. literalinclude:: ../../../flash_examples/predict/object_detection.py
    :language: python
    :lines: 14-

For more advanced inference options, see :ref:`predictions`.

------

**********
Finetuning
**********

To tailor the object detector to your dataset, you would need to have it in `COCO Format <https://cocodataset.org/#format-data>`_, and then finetune the model.

.. tip:: You could also pass `trainable_backbone_layers` to :class:`~flash.vision.ObjectDetector` and train the model.

.. literalinclude:: ../../../flash_examples/finetuning/object_detection.py
    :language: python
    :lines: 14-

------

*****
Model
*****

By default, we use the `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_ model with a ResNet-50 FPN backbone.
We also support `RetinaNet <https://arxiv.org/abs/1708.02002>`_.
The inputs could be images of different sizes.
The model behaves differently for training and evaluation.
For training, it expects both the input tensors as well as the targets. And during the evaluation, it expects only the input tensors and returns predictions for each image.
The predictions are a list of boxes, labels, and scores.

------

*********************
Changing the backbone
*********************
By default, we use a ResNet-50 FPN backbone. You can change the backbone for the model by passing in a different backbone.

.. testsetup::

    from flash.data.utils import download_data
    from flash.vision import ObjectDetectionData, ObjectDetector

    download_data("https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip", "data/")

.. testcode::

    # 1. Organize the data
    datamodule = ObjectDetectionData.from_coco(
        train_folder="data/coco128/images/train2017/",
        train_ann_file="data/coco128/annotations/instances_train2017.json",
        batch_size=2
    )

    # 2. Build the Task
    model = ObjectDetector(model="retinanet", backbone="resnet101", num_classes=datamodule.num_classes)

.. testoutput::
    :hide:

    ...

.. include:: ../common/object_detection_backbones.rst

------

*************
API reference
*************

.. _object_detector:

ObjectDetector
--------------

.. autoclass:: flash.vision.ObjectDetector
    :members:
    :exclude-members: forward

.. _object_detection_data:

ObjectDetectionData
-------------------

.. autoclass:: flash.vision.ObjectDetectionData

.. automethod:: flash.vision.ObjectDetectionData.from_coco
