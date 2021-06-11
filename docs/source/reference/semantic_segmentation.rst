
.. _semantic_segmentation:

######################
Semantic Segmentation
######################

********
The task
********
Semantic Segmentation, or image segmentation, is the task of performing classification at a pixel-level, meaning each pixel will associated to a given class. The model output shape is ``(batch_size, num_classes, heigh, width)``.

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

A :class:`~flash.image.SemanticSegmentation` `fcn_resnet50` pre-trained on `CARLA <http://carla.org/>`_ simulator is provided for the inference example.


Use the :class:`~flash.image.SemanticSegmentation` pretrained model for inference on any string sequence using :func:`~flash.image.SemanticSegmentation.predict`:

.. literalinclude:: ../../../flash_examples/predict/semantic_segmentation.py
    :language: python
    :lines: 14-

For more advanced inference options, see :ref:`predictions`.

------

**********
Finetuning
**********

you now want to customise your model with new data using the same dataset.
Once we download the data using :func:`~flash.core.data.download_data`, all we need is the train data and validation data folders to create the :class:`~flash.image.SemanticSegmentationData`.

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


Now all we need is to train our task!

.. literalinclude:: ../../../flash_examples/finetuning/semantic_segmentation.py
    :language: python
    :lines: 14-

------

*************
API reference
*************

.. _segmentation:

SemanticSegmentation
--------------------

.. autoclass:: flash.image.SemanticSegmentation
    :members:
    :exclude-members: forward

.. _segmentation_data:

SemanticSegmentationData
------------------------

.. autoclass:: flash.image.SemanticSegmentationData

.. automethod:: flash.image.SemanticSegmentationData.from_folders

.. autoclass:: flash.image.SemanticSegmentationPreprocess
