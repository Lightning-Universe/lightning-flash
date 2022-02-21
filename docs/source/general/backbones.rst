.. _backbones_heads:

*******************
Backbones and Heads
*******************

Backbones are the pre trained models that can be used with a task.
The backbones or heads that are available can be found by using the ``available_backbones`` and ``available_heads`` methods.

To get the available backbones for a task like :class:`~flash.image.classification.model.ImageClassifier`, run:

.. code-block:: python

    from flash.image import ImageClassifier

    # get the backbones available for ImageClassifier
    backbones = ImageClassifier.available_backbones()

    # print the backbones
    print(backbones)

To get the available heads for a task like :class:`~flash.image.segmentation.model.SemanticSegmentation`, run:

.. code-block:: python

    from flash.image import SemanticSegmentation

    # get the heads available for SemanticSegmentation
    heads = SemanticSegmentation.available_heads()

    # print the heads
    print(heads)
