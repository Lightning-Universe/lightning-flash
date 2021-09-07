**********
Backbones
**********

Backbones are the pre trained models that can be used to fine tune a task. 
The backbones that are available can be found by using the :func:`Task.available_backbones`.

To get the available backbones for a task like :class:`~flash.image.classification.model.ImageClassifier`, run:

.. code-block:: python

    from flash.image import ImageClassifier

    # get the backbones available for ImageClassifier
    backbones = ImageClassifier.available_backbones()

    # print the backbones
    print(backbones)

