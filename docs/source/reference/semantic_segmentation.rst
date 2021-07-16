
.. _semantic_segmentation:

#####################
Semantic Segmentation
#####################

********
The Task
********
Semantic Segmentation, or image segmentation, is the task of performing classification at a pixel-level, meaning each pixel will associated to a given class.
See more: https://paperswithcode.com/task/semantic-segmentation

------

*******
Example
*******

Let's look at an example using a data set generated with the `CARLA <http://carla.org/>`_ driving simulator.
The data was generated as part of the `Kaggle Lyft Udacity Challenge <https://www.kaggle.com/kumaresanmanickavelu/lyft-udacity-challenge>`_.
The data contains one folder of images and another folder with the corresponding segmentation masks.
Here's the structure:

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

Once we've downloaded the data using :func:`~flash.core.data.download_data`, we create the :class:`~flash.image.segmentation.data.SemanticSegmentationData`.
We select a pre-trained ``mobilenet_v3_large`` backbone with an ``fpn`` head to use for our :class:`~flash.image.segmentation.model.SemanticSegmentation` task and fine-tune on the CARLA data.
We then use the trained :class:`~flash.image.segmentation.model.SemanticSegmentation` for inference. You can check the available pretrained weights for the backbones like this `SemanticSegmentation.available_pretrained_weights("resnet18")`.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/semantic_segmentation.py
    :language: python
    :lines: 14-

------

*******
Serving
*******

The :class:`~flash.image.segmentation.model.SemanticSegmentation` task is servable.
This means you can call ``.serve`` to serve your :class:`~flash.core.model.Task`.
Here's an example:

.. literalinclude:: ../../../flash_examples/serve/semantic_segmentation/inference_server.py
    :language: python
    :lines: 14-

You can now perform inference from your client like this:

.. literalinclude:: ../../../flash_examples/serve/semantic_segmentation/client.py
    :language: python
    :lines: 14-
