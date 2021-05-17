.. _contributing_optional:

***************
Optional Extras
***************

transforms.py
=============

Sometimes you'd like to have quite a few transforms by default (standard augmentations, normalization, etc.).
If you do then, for better organization, you can define a ``transforms.py`` which houses your default transforms to be referenced in your :class:`~flash.core.data.process.Preprocess`.
Here's an example from ``image/classification/transforms.py`` which creates some default transforms given the desired image size:

.. literalinclude:: ../../../flash/image/classification/transforms.py
    :language: python
    :pyobject: default_transforms

Here's how we create our transforms in the :class:`~flash.image.classification.data.ImageClassificationPreprocess`:

.. literalinclude:: ../../../flash/image/classification/data.py
    :language: python
    :pyobject: ImageClassificationPreprocess.default_transforms

backbones.py
============

In Flash, we love to provide as much access to the state-of-the-art as we can.
To this end, we've created the :any:`FlashRegistry <registry>`.
The registry allows you to register backbones for your task that can be selected by the user.
The backbones can come from anywhere as long as you can register a function that loads the backbone.
If you want to configure some backbones for your task, it's best practice to include these in a ``backbones.py`` file.
Here's an example adding ``SimCLR`` to the ``IMAGE_CLASSIFIER_BACKBONES``, from ``image/backbones.py``:

.. literalinclude:: ../../../flash/image/backbones.py
    :language: python
    :pyobject: load_simclr_imagenet

In ``image/classification/model.py``, we attach ``IMAGE_CLASSIFIER_BACKBONES`` to the :class:`~flash.image.classification.model.ImageClassifier` as a class attribute ``backbones``.
Now we get the backbone from the registry and create a head in the ``__init__``:

.. literalinclude:: ../../../flash/image/classification/model.py
    :language: python
    :pyobject: ImageClassifier.__init__

Finally, we use our backbone and head in a custom forward pass:

.. literalinclude:: ../../../flash/image/classification/model.py
    :language: python
    :pyobject: ImageClassifier.forward

serialization.py
================

Sometimes you want to give the user some control over their prediction format.
`Postprocess` can do the heavy lifting (anything you always want to apply to the predictions), but one or more custom `Serializer` implementations can be used to convert the predictions to a desired output format.
A good example is in classification; sometimes we'd like the classes, sometimes the logits, sometimes the labels, you get the idea.
You should add your `Serializer` implementations in a `serialization.py` file and set a good default in your `Task`.
