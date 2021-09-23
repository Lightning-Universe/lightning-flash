.. customcarditem::
   :header: Instance Segmentation
   :card_description: Learn to segment objects in images with Flash and build a model for segmenting images of pets.
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/instance_segmentation.svg
   :tags: Image,Segmentation,Detection

.. _instance_segmentation:

#####################
Instance Segmentation
#####################

********
The Task
********

Instance segmentation is the task of segmenting objects images and determining their associated classes.

The :class:`~flash.image.instance_segmentation.model.InstanceSegmentation` and :class:`~flash.image.instance_segmentation.data.InstanceSegmentationData` classes internally rely on `IceVision <https://airctic.com/>`_.

------

*******
Example
*******

Let's look at instance segmentation with `The Oxford-IIIT Pet Dataset <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_ from `IceData <https://github.com/airctic/icedata>`_.
Once we've downloaded the data, we can create the :class:`~flash.image.instance_segmentation.data.InstanceSegmentationData`.
We select a ``mask_rcnn`` with a ``resnet18_fpn`` backbone to use for our :class:`~flash.image.instance_segmentation.model.InstanceSegmentation` and fine-tune on the pets data.
We then use the trained :class:`~flash.image.instance_segmentation.model.InstanceSegmentation` for inference.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/instance_segmentation.py
    :language: python
    :lines: 14-

------

**********
Flash Zero
**********

The instance segmentation task can be used directly from the command line with zero code using :ref:`flash_zero`.
You can run the above example with:

.. code-block:: bash

    flash instance_segmentation

To view configuration options and options for running the instance segmentation task with your own data, use:

.. code-block:: bash

    flash instance_segmentation --help
