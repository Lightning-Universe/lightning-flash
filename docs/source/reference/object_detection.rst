
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

.. code-block:: python

	from flash.vision import ObjectDetector

	# 1. Load the model
	detector = ObjectDetector()

	# 2. Perform inference on an image file
	predictions = detector.predict("path/to/image.png")
	print(predictions)

Or on a random image tensor

.. code-block:: python

    # Perform inference on a random image tensor
    import torch
    images = torch.rand(32, 3, 1080, 1920)
    predictions = detector.predict(images)
    print(predictions)

For more advanced inference options, see :ref:`predictions`.

------

**********
Finetuning
**********

To tailor the object detector to your dataset, you would need to have it in `COCO Format <https://cocodataset.org/#format-data>`_, and then finetune the model.

.. code-block:: python

    import flash
    from flash.core.data import download_data
    from flash.vision import ObjectDetectionData, ObjectDetector

    # 1. Download the data
    # Dataset Credit: https://www.kaggle.com/ultralytics/coco128
    download_data("https://github.com/zhiqwang/yolov5-rt-stack/releases/download/v0.3.0/coco128.zip", "data/")

    # 2. Load the Data
    datamodule = ObjectDetectionData.from_coco(
        train_folder="data/coco128/images/train2017/",
        train_ann_file="data/coco128/annotations/instances_train2017.json",
        batch_size=2
    )

    # 3. Build the model
    model = ObjectDetector(num_classes=datamodule.num_classes)

    # 4. Create the trainer. Run thrice on data
    trainer = flash.Trainer(max_epochs=3)

    # 5. Finetune the model
    trainer.finetune(model, datamodule)

    # 6. Save it!
    trainer.save_checkpoint("object_detection_model.pt")

------

*****
Model
*****

By default, we use the `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_ model with a ResNet-50 FPN backbone. The inputs could be images of different sizes. The model behaves differently for training and evaluation. For training, it expects both the input tensors as well as the targets. And during evaluation, it expects only the input tensors and returns predictions for each image. The predictions are a list of boxes, labels and scores.

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
